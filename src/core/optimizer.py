"""Single-trajectory antibody optimization."""

from __future__ import annotations

import glob
import logging
import os
import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

from src.core.candidate_pool import limit_embedding_pool
from src.core.pareto_front import select_top_k_pareto
from src.data.database import BindSculptDatabase
from src.models.esm_engine import ESMEngine
from src.models.esmfold2_oracle import ESMFold2Oracle
from src.models.mpnn_mutator import MPNNMutator
from src.models.surrogate_mobo import SurrogateMOBO
from src.utils.cdr import (
    AntibodySequenceError,
    get_antibody_cdr_indices,
    validate_antibody_sequence,
)
from src.utils.geometry import HotspotGeometry

logger = logging.getLogger(__name__)


class SingleTrajectoryOptimizer:
    def __init__(self, config):
        self.config = config
        self.run_dir = config.paths.run_output_dir
        self.db = BindSculptDatabase(self.run_dir)
        self.esm_engine = ESMEngine(config)
        self.esmfold2_oracle = ESMFold2Oracle(config)
        self.final_oracle = ESMFold2Oracle(
            config,
            variant=config.models.esmfold2.get("final_variant", "full"),
        )
        self.mpnn_mutator = MPNNMutator(config)
        self.surrogate = SurrogateMOBO(config)
        self.geometry = HotspotGeometry()

        self.max_iterations = int(
            config.active_learning.get("num_iterations", 5)
        )
        self.scaffold_dir = config.paths.get(
            "scaffold_dir", "data/inputs/scaffolds"
        )
        antibody = config.get("antibody", {})
        chain_types = antibody.get("chain_types", {"B": "VHH"})
        self.chain_types = list(chain_types.values())
        self.linker = antibody.get("linker", "GGGGSGGGGSGGGGS")
        self.seed = int(config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _cdr_positions(self, binder):
        positions = binder.get("cdr_positions")
        if isinstance(positions, list) and positions:
            return [int(position) for position in positions]
        positions = get_antibody_cdr_indices(
            binder["binder_sequence"], self.chain_types, self.linker
        )
        binder["cdr_positions"] = positions
        return positions

    def _valid_antibody(self, binder):
        try:
            sequence = validate_antibody_sequence(
                binder["binder_sequence"], self.chain_types, self.linker
            )
            positions = binder.get("cdr_positions")
            if isinstance(positions, list) and positions:
                positions = [int(position) for position in positions]
                if min(positions) < 0 or max(positions) >= len(sequence):
                    raise AntibodySequenceError(
                        "CDR position is outside the sequence"
                    )
                binder["cdr_positions"] = positions
            else:
                binder["cdr_positions"] = get_antibody_cdr_indices(
                    sequence, self.chain_types, self.linker
                )
            return True
        except (
            KeyError, TypeError, ValueError, AntibodySequenceError
        ) as exc:
            binder["filter_reason"] = f"invalid_antibody: {exc}"
            return False

    def _meets_filters(self, binder):
        filters = self.config.filters.complex_filters
        return (
            binder.get("esmfold2_status") == "success"
            and not binder.get("filter_reason")
            and binder.get("ipsae_min", 0)
            >= filters.get("min_ipsae", 0.6)
            and binder.get("iptm", 0) >= filters.get("min_iptm", 0.8)
            and binder.get("binder_plddt", 0)
            >= filters.get("min_binder_plddt", 0.8)
        )

    def _score_and_penalize(self, binders, hotspot_str):
        hotspot_cfg = self.config.get("hotspot", {})
        for binder in binders:
            binder["composite_affinity"] = 0.0
            binder["hotspot_satisfied"] = False
            binder.pop("filter_reason", None)
            if binder.get("esmfold2_status") != "success":
                continue
            if not self._valid_antibody(binder):
                continue

            metrics = self.geometry.hotspot_contact_metrics(
                binder["complex_pdb_path"],
                hotspot_str or "",
                self._cdr_positions(binder),
                distance_threshold=float(
                    hotspot_cfg.get("contact_distance", 5.0)
                ),
            )
            binder.update(metrics)
            cdr_pass = (
                metrics["cdr_hotspot_contacts"]
                >= hotspot_cfg.get("min_cdr_contacts", 3)
                and metrics["cdr_contact_fraction"]
                >= hotspot_cfg.get("min_cdr_fraction", 0.5)
            )
            hotspot_pass = (
                not hotspot_str
                or (
                    metrics["hotspot_residue_contacts"]
                    >= hotspot_cfg.get("min_hotspot_residues", 3)
                    and metrics["hotspot_contact_fraction"]
                    >= hotspot_cfg.get("min_hotspot_fraction", 0.5)
                )
            )
            if not cdr_pass or not hotspot_pass:
                binder["filter_reason"] = (
                    "failed_hotspot_contacts"
                    if hotspot_str else "failed_cdr_interface"
                )
                continue

            binder["hotspot_satisfied"] = True
            binder["composite_affinity"] = (
                0.5 * binder.get("ipsae_min", 0.0)
                + 0.3 * binder.get("iptm", 0.0)
                + 0.2 * binder.get("binder_plddt", 0.0)
            )
        return binders

    def _generate_comprehensive_mutants(
        self, current_best, max_mutants=200
    ):
        """Generate CDR-only single mutants, then unique CDR double mutants."""
        base_sequence = current_best["binder_sequence"]
        mutable = [
            index
            for index in self._cdr_positions(current_best)
            if base_sequence[index] != "C"
        ]
        alphabet = "ADEFGHIKLMNPQRSTVWY"
        parent_id = current_best["id"]
        mutants = []

        for position in mutable:
            original = base_sequence[position]
            for residue in alphabet:
                if residue == original:
                    continue
                mutants.append({
                    "id": (
                        f"single_{original}{position + 1}{residue}_"
                        f"{uuid.uuid4().hex[:4]}"
                    ),
                    "binder_sequence": (
                        base_sequence[:position]
                        + residue
                        + base_sequence[position + 1:]
                    ),
                    "mutation_type": "cdr_single",
                    "parent_id": parent_id,
                    "cdr_positions": list(self._cdr_positions(current_best)),
                })

        random.shuffle(mutants)
        mutants = mutants[:max_mutants]
        seen = {mutant["binder_sequence"] for mutant in mutants}
        attempts = 0
        while len(mutants) < max_mutants and len(mutable) >= 2:
            attempts += 1
            if attempts > max_mutants * 20:
                break
            first, second = random.sample(mutable, 2)
            sequence = list(base_sequence)
            for position in (first, second):
                choices = [
                    residue for residue in alphabet
                    if residue != sequence[position]
                ]
                sequence[position] = random.choice(choices)
            sequence = "".join(sequence)
            if sequence in seen:
                continue
            seen.add(sequence)
            mutants.append({
                "id": f"double_{uuid.uuid4().hex[:8]}",
                "binder_sequence": sequence,
                "mutation_type": "cdr_double",
                "parent_id": parent_id,
                "cdr_positions": list(self._cdr_positions(current_best)),
            })
        logger.info("Generated %d CDR-only point mutants.", len(mutants))
        return mutants

    @staticmethod
    def _delta_rmsd(bound_path, monomer_path):
        try:
            parser = PDBParser(QUIET=True)
            bound = parser.get_structure("bound", bound_path)
            monomer = parser.get_structure("monomer", monomer_path)
            bound_ca = [
                atom for atom in bound[0]["B"].get_atoms()
                if atom.name == "CA"
            ]
            monomer_ca = [
                atom for atom in monomer[0]["A"].get_atoms()
                if atom.name == "CA"
            ]
            if len(bound_ca) != len(monomer_ca) or not bound_ca:
                return float("inf")
            superimposer = Superimposer()
            superimposer.set_atoms(bound_ca, monomer_ca)
            return float(superimposer.rms)
        except Exception as exc:
            logger.warning("Delta RMSD failed: %s", exc)
            return float("inf")

    def _write_final(self, candidates):
        output = Path(self.run_dir) / "final_post_processed_candidates.csv"
        if candidates:
            frame = pd.DataFrame(candidates).sort_values(
                ["composite_affinity", "delta_rmsd"],
                ascending=[False, True],
            )
        else:
            frame = pd.DataFrame(columns=[
                "id",
                "binder_sequence",
                "original_parent",
                "ipsae_min",
                "iptm",
                "binder_plddt",
                "composite_affinity",
                "delta_rmsd",
                "complex_pdb_path",
            ])
        frame.to_csv(output, index=False)
        return frame

    def _final_mpnn_polish_and_evaluate(
        self,
        target_id,
        target_seq,
        hotspot_str,
        top_k=2,
        generate_per_candidate=100,
        keep_per_candidate=5,
    ):
        filters = self.config.filters.complex_filters
        parents = [
            binder for binder in self.db.get_all_training_data()
            if binder.get("esmfold2_status") == "success"
            and binder.get("composite_affinity", 0) > 0
            and not binder.get("filter_reason")
        ]
        parents.sort(
            key=lambda binder: binder.get("composite_affinity", 0),
            reverse=True,
        )
        parents = parents[:top_k]
        if not parents:
            logger.warning("No parent met strict filters for final AbMPNN redesign.")
            self._write_final([])
            return None

        old_threshold = self.mpnn_mutator.config.fixed_pos_distance_threshold
        old_temperature = self.mpnn_mutator.config.sampling_temp
        polished = []
        try:
            self.mpnn_mutator.config.fixed_pos_distance_threshold = float(
                filters.get("final_fixed_distance", 4.0)
            )
            self.mpnn_mutator.config.sampling_temp = float(
                filters.get("final_sampling_temp", 0.1)
            )
            for parent in parents:
                pdb_path = parent.get("complex_pdb_path")
                if not isinstance(pdb_path, str) or not os.path.exists(pdb_path):
                    continue
                mutants = self.mpnn_mutator.generate_mutants(
                    pdb_path, generate_per_candidate
                )
                mutants.sort(
                    key=lambda mutant: mutant.get("mpnn_score", float("inf"))
                )
                kept = 0
                for mutant in mutants:
                    if mutant["binder_sequence"] == parent["binder_sequence"]:
                        continue
                    mutant.update({
                        "target_id": target_id,
                        "target_sequence": target_seq,
                        "mutation_type": "final_abmpnn_redesign",
                        "original_parent": parent["id"],
                    })
                    polished.append(mutant)
                    kept += 1
                    if kept >= keep_per_candidate:
                        break
        finally:
            self.mpnn_mutator.config.fixed_pos_distance_threshold = old_threshold
            self.mpnn_mutator.config.sampling_temp = old_temperature

        if not polished:
            logger.warning("Final AbMPNN produced no valid redesigned sequences.")
            self._write_final([])
            return None

        screened = self.esmfold2_oracle.predict_batch(
            polished, mode="complex"
        )
        screened = self._score_and_penalize(screened, hotspot_str)
        screened = [
            binder for binder in screened
            if binder.get("composite_affinity", 0) > 0
        ]
        screened.sort(
            key=lambda binder: binder["composite_affinity"], reverse=True
        )
        shortlist = screened[
            : int(self.config.active_learning.get(
                "final_rescore_candidates", 10
            ))
        ]
        if not shortlist:
            logger.warning("No final AbMPNN redesign passed Fast screening.")
            self._write_final([])
            return None

        seed_count = int(self.config.models.esmfold2.get(
            "final_num_seeds", 5
        ))
        evaluated = self.final_oracle.predict_batch(
            shortlist,
            mode="complex",
            seeds=range(self.seed, self.seed + seed_count),
        )
        evaluated = self._score_and_penalize(evaluated, hotspot_str)
        self.db.save_evaluated_batch(evaluated)
        survivors = [
            binder for binder in evaluated if self._meets_filters(binder)
        ]
        if not survivors:
            logger.warning("No Full-rescored AbMPNN redesign passed filters.")
            self._write_final([])
            return None

        monomers = self.final_oracle.predict_batch([
            {
                "id": f"monomer_{binder['id']}",
                "binder_sequence": binder["binder_sequence"],
            }
            for binder in survivors
        ], mode="monomer")
        monomer_by_id = {
            result["id"]: result for result in monomers
        }
        max_rmsd = float(filters.get("max_delta_rmsd", 2.0))
        final = []
        for binder in survivors:
            monomer = monomer_by_id.get(f"monomer_{binder['id']}", {})
            if monomer.get("esmfold2_status") != "success":
                continue
            binder["delta_rmsd"] = self._delta_rmsd(
                binder["complex_pdb_path"],
                monomer["complex_pdb_path"],
            )
            if binder["delta_rmsd"] <= max_rmsd:
                final.append(binder)

        frame = self._write_final(final)
        if frame.empty:
            logger.warning("No final AbMPNN redesign passed delta RMSD.")
            return None
        logger.info("Saved %d strict final candidates.", len(frame))
        return frame.iloc[0].to_dict()

    def optimize_binder(
        self,
        target_id,
        target_seq,
        target_pdb,
        hotspot_residues_str,
    ):
        logger.info("Starting antibody optimization for %s", target_id)
        self._write_final([])
        active = self.config.active_learning
        self.esm_engine.load_model()
        self.esm_engine.cache_target_embedding(target_seq)

        raw_scaffolds = (
            sorted(glob.glob(os.path.join(self.scaffold_dir, "*.pdb")))
            + sorted(glob.glob(os.path.join(self.scaffold_dir, "*.cif")))
        )
        scaffold_files = []
        seen_stems = set()
        for path in raw_scaffolds:
            if Path(path).stem not in seen_stems:
                seen_stems.add(Path(path).stem)
                scaffold_files.append(path)
        random.shuffle(scaffold_files)
        if not scaffold_files:
            raise FileNotFoundError(
                f"No scaffolds found in {self.scaffold_dir}"
            )

        pose_dir = os.path.join(self.run_dir, "initial_poses")
        poses = []
        accepted_scaffolds = 0
        for scaffold_path in scaffold_files:
            cdr_indices = self.mpnn_mutator.get_scaffold_cdr_indices(
                scaffold_path
            )
            scaffold_poses = self.geometry.place_scaffolds_diverse(
                target_pdb=target_pdb,
                scaffold_pdb=scaffold_path,
                out_dir=pose_dir,
                prefix=Path(scaffold_path).stem,
                hotspot_residues_str=hotspot_residues_str,
                num_poses=int(active.get("poses_per_scaffold", 5)),
                max_attempts=int(active.get("placement_attempts", 1000)),
                cdr_indices_by_chain=cdr_indices,
            )
            if scaffold_poses:
                poses.extend(scaffold_poses)
                accepted_scaffolds += 1
            if accepted_scaffolds >= int(active.get("scaffolds_to_accept", 3)):
                break
        if not poses:
            raise RuntimeError(
                "No scaffold could be placed at the target surface"
            )

        old_threshold = self.mpnn_mutator.config.fixed_pos_distance_threshold
        try:
            self.mpnn_mutator.config.fixed_pos_distance_threshold = -1.0
            batches = self.mpnn_mutator.generate_mutants_batch(
                poses,
                num_mutants=int(active.get("initial_mpnn_mutants", 100)),
            )
        finally:
            self.mpnn_mutator.config.fixed_pos_distance_threshold = old_threshold

        initial = []
        seen = set()
        for mutants in batches:
            for mutant in sorted(
                mutants, key=lambda item: item.get("mpnn_score", float("inf"))
            ):
                if mutant["binder_sequence"] in seen:
                    continue
                seen.add(mutant["binder_sequence"])
                mutant.update({
                    "target_id": target_id,
                    "target_sequence": target_seq,
                    "iteration": 0,
                })
                initial.append(mutant)
                break
        if not initial:
            raise RuntimeError("AbMPNN produced no valid initial antibodies")

        evaluated = self.esmfold2_oracle.predict_batch(initial, mode="complex")
        evaluated = self._score_and_penalize(
            evaluated, hotspot_residues_str
        )
        evaluated = self.esm_engine.get_concat_embeddings(evaluated)
        self.db.save_evaluated_batch(evaluated)
        successful = [
            binder for binder in evaluated
            if binder.get("composite_affinity", 0) > 0
        ]
        if not successful:
            logger.error("All initial antibodies failed prediction or hotspot checks.")
            self._write_final([])
            return None
        current_best = max(
            successful, key=lambda binder: binder["composite_affinity"]
        )

        original_threshold = (
            self.mpnn_mutator.config.fixed_pos_distance_threshold
        )
        stalled = False
        patience = 0
        max_patience = int(
            self.config.active_learning.get("max_patience", 3)
        )
        for iteration in range(1, self.max_iterations + 1):
            historical = self.db.get_all_training_data()
            self.surrogate.train(historical)
            min_iptm = self.config.filters.complex_filters.get(
                "min_iptm", 0.8
            )
            if current_best.get("iptm", 0) >= min_iptm:
                mutants = self._generate_comprehensive_mutants(
                    current_best,
                    int(self.config.active_learning.get(
                        "point_mutants", 200
                    )),
                )
            else:
                self.mpnn_mutator.config.fixed_pos_distance_threshold = (
                    -1.0
                    if current_best.get("iptm", 0) < 0.7 or stalled
                    else original_threshold
                )
                mutants = self.mpnn_mutator.generate_mutants(
                    current_best["complex_pdb_path"],
                    int(self.config.active_learning.get(
                        "mpnn_mutants", 200
                    )),
                )
                for mutant in mutants:
                    mutant["parent_id"] = current_best["id"]

            seen = self.db.get_seen_sequences()
            unique = []
            for mutant in mutants:
                if mutant["binder_sequence"] in seen:
                    continue
                seen.add(mutant["binder_sequence"])
                mutant.update({
                    "target_id": target_id,
                    "target_sequence": target_seq,
                    "iteration": iteration,
                })
                unique.append(mutant)
            if not unique:
                logger.warning("Mutation pool was exhausted.")
                break

            original_count = len(unique)
            unique = limit_embedding_pool(
                unique,
                self.config.active_learning.get("embedding_pool_size", 200),
            )
            logger.info(
                "Embedding %d/%d unique mutants.",
                len(unique), original_count,
            )
            unique = self.esm_engine.get_concat_embeddings(unique)
            unique = self.surrogate.predict(unique)
            selected = select_top_k_pareto(
                unique,
                int(self.config.active_learning.candidates_per_iteration),
            )
            evaluated = self.esmfold2_oracle.predict_batch(
                selected, mode="complex"
            )
            evaluated = self._score_and_penalize(
                evaluated, hotspot_residues_str
            )
            self.db.save_evaluated_batch(evaluated)
            iteration_successes = [
                binder for binder in evaluated
                if binder.get("composite_affinity", 0) > 0
            ]
            iteration_best = max(
                iteration_successes,
                key=lambda binder: binder["composite_affinity"],
                default=None,
            )
            if (
                iteration_best
                and iteration_best["composite_affinity"]
                > current_best["composite_affinity"]
            ):
                current_best = iteration_best
                stalled = False
                patience = 0
            else:
                stalled = True
                patience += 1
                if patience >= max_patience:
                    logger.warning(
                        "Stopping after %d stalled iterations.", patience
                    )
                    break

        self.mpnn_mutator.config.fixed_pos_distance_threshold = original_threshold
        self.surrogate.save_models(prefix=f"target_{target_id}_final")
        final_best = self._final_mpnn_polish_and_evaluate(
            target_id,
            target_seq,
            hotspot_residues_str,
            top_k=int(active.get("final_parent_count", 2)),
            generate_per_candidate=int(active.get("final_mpnn_mutants", 100)),
            keep_per_candidate=int(active.get("final_keep_per_parent", 5)),
        )
        if final_best:
            logger.info(
                "Optimization finished with strict final affinity %.3f",
                final_best["composite_affinity"],
            )
        return final_best
