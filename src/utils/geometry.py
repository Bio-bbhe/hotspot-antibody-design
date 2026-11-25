"""Antibody scaffold placement and hotspot-contact checks."""

from __future__ import annotations

import copy
import logging
import os

import numpy as np
from Bio.PDB import MMCIFParser, Model, PDBIO, PDBParser, Structure
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class HotspotGeometry:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()

    @staticmethod
    def _coords(chain, atom_names=None):
        atom_names = atom_names or {"CA", "C", "N", "O", "CB"}
        return np.asarray([
            atom.coord
            for residue in chain
            if residue.id[0] == " "
            for atom in residue
            if atom.name in atom_names
        ])

    def _load_structure(self, name, file_path):
        parser = MMCIFParser(QUIET=True) if str(file_path).lower().endswith(".cif") else self.parser
        return parser.get_structure(name, str(file_path))

    @staticmethod
    def _parse_residue_ranges(value):
        residues = []
        for block in value.split(","):
            block = block.strip()
            if not block:
                continue
            if "-" in block:
                start, end = map(int, block.split("-", 1))
                residues.extend(range(start, end + 1))
            else:
                residues.append(int(block))
        return sorted(set(residues))

    @staticmethod
    def _sequential_to_pdb_resnums(structure, chain_id, positions):
        residue_numbers = [
            residue.id[1]
            for residue in structure[0][chain_id]
            if residue.id[0] == " "
        ]
        if residue_numbers == list(range(1, len(residue_numbers) + 1)):
            return positions
        return [
            residue_numbers[position - 1]
            for position in positions
            if 1 <= position <= len(residue_numbers)
        ]

    @staticmethod
    def _residue_com(chain, residue_numbers):
        coords = [
            residue["CA"].coord
            for residue in chain
            if residue.id[0] == " "
            and residue.id[1] in residue_numbers
            and "CA" in residue
        ]
        if not coords:
            raise ValueError("Configured hotspot residues were not found")
        return np.mean(coords, axis=0)

    @staticmethod
    def _paratope_coords(chains, cdr_indices_by_chain):
        if not cdr_indices_by_chain:
            return None
        if len(chains) != len(cdr_indices_by_chain):
            raise ValueError("CDR chain count does not match scaffold chain count")
        coords = []
        for chain, indices in zip(chains, cdr_indices_by_chain):
            residues = [
                residue for residue in chain if residue.id[0] == " "
            ]
            for index in indices:
                if index < len(residues):
                    coords.extend(
                        atom.coord for atom in residues[index]
                        if atom.element != "H"
                    )
        return np.asarray(coords) if coords else None

    @staticmethod
    def _oriented_rotation(scaffold_com, paratope_com, inward_vector):
        source = paratope_com - scaffold_com
        if np.linalg.norm(source) < 1e-6:
            return Rotation.random().as_matrix()
        source /= np.linalg.norm(source)
        alignment, _ = Rotation.align_vectors([inward_vector], [source])
        twist = Rotation.from_rotvec(
            -inward_vector * np.random.uniform(0.0, 2.0 * np.pi)
        )
        tilt = Rotation.from_rotvec(np.random.normal(0.0, 0.12, size=3))
        return (tilt * twist * alignment).as_matrix()

    def place_scaffolds_diverse(
        self,
        target_pdb,
        scaffold_pdb,
        out_dir,
        prefix,
        hotspot_residues_str,
        target_chain_id="A",
        scaffold_chain_id="B",
        num_poses=10,
        max_attempts=5000,
        clash_cutoff=2.8,
        min_contacts=15,
        contact_dist=6.0,
        rmsd_threshold=4.0,
        cdr_indices_by_chain=None,
    ):
        """Generate diverse poses, orienting CDRs toward a configured hotspot."""
        os.makedirs(out_dir, exist_ok=True)
        target_structure = self._load_structure("target", target_pdb)
        scaffold_structure = self._load_structure("scaffold", scaffold_pdb)
        target_chain = target_structure[0][target_chain_id]

        scaffold_chains = []
        for index, chain in enumerate(scaffold_structure[0].get_chains()):
            if not any(True for _ in chain.get_atoms()):
                continue
            chain_copy = copy.deepcopy(chain)
            chain_copy.id = chr(ord(scaffold_chain_id) + index)
            scaffold_chains.append(chain_copy)
        if not scaffold_chains:
            return []

        scaffold_bb = np.concatenate([
            self._coords(chain) for chain in scaffold_chains
        ])
        scaffold_ca = np.concatenate([
            self._coords(chain, {"CA"}) for chain in scaffold_chains
        ])
        scaffold_com = scaffold_bb.mean(axis=0)
        paratope = self._paratope_coords(
            scaffold_chains, cdr_indices_by_chain
        )
        paratope_com = paratope.mean(axis=0) if paratope is not None else scaffold_com

        target_bb = self._coords(target_chain)
        target_tree = cKDTree(target_bb)
        target_com = target_bb.mean(axis=0)

        hotspot = bool(hotspot_residues_str)
        if hotspot:
            positions = self._parse_residue_ranges(hotspot_residues_str)
            positions = self._sequential_to_pdb_resnums(
                target_structure, target_chain_id, positions
            )
            hotspot_com = self._residue_com(target_chain, positions)
            outward = hotspot_com - target_com
            outward /= np.linalg.norm(outward) or 1.0
            inward = -outward

        saved_paths = []
        saved_ca = []
        for _attempt in range(max_attempts):
            if len(saved_paths) >= num_poses:
                break

            if hotspot:
                rotation = self._oriented_rotation(
                    scaffold_com, paratope_com, inward
                )
                tangent = np.random.normal(size=3)
                tangent -= tangent.dot(outward) * outward
                tangent /= np.linalg.norm(tangent) or 1.0
                desired_paratope = (
                    hotspot_com
                    + outward * np.random.uniform(6.0, 9.0)
                    + tangent * np.random.uniform(-3.0, 3.0)
                )
                translation = desired_paratope - paratope_com @ rotation.T
            else:
                surface = target_bb[np.random.randint(len(target_bb))]
                outward = surface - target_com
                if np.linalg.norm(outward) < 1e-6:
                    continue
                outward /= np.linalg.norm(outward)
                rotation = self._oriented_rotation(
                    scaffold_com, paratope_com, -outward
                )
                tangent = np.random.normal(size=3)
                tangent -= tangent.dot(outward) * outward
                tangent /= np.linalg.norm(tangent) or 1.0
                desired_paratope = (
                    surface
                    + outward * np.random.uniform(6.0, 9.0)
                    + tangent * np.random.uniform(-3.0, 3.0)
                )
                translation = desired_paratope - paratope_com @ rotation.T

            transformed_bb = scaffold_bb @ rotation.T + translation
            distances, _ = target_tree.query(transformed_bb, k=1)
            if np.any(distances < clash_cutoff):
                continue
            if np.sum(distances < contact_dist) < min_contacts:
                continue

            transformed_ca = scaffold_ca @ rotation.T + translation
            if any(
                np.sqrt(np.mean(np.sum((transformed_ca - prior) ** 2, axis=1)))
                < rmsd_threshold
                for prior in saved_ca
            ):
                continue
            saved_ca.append(transformed_ca)

            merged = Structure.Structure(f"merged_{len(saved_paths)}")
            model = Model.Model(0)
            model.add(copy.deepcopy(target_chain))
            for chain in scaffold_chains:
                transformed_chain = copy.deepcopy(chain)
                for atom in transformed_chain.get_atoms():
                    atom.coord = atom.coord @ rotation.T + translation
                model.add(transformed_chain)
            merged.add(model)

            output = os.path.join(
                out_dir, f"{prefix}_pose_{len(saved_paths)}.pdb"
            )
            self.io.set_structure(merged)
            self.io.save(output)
            saved_paths.append(output)

        if not saved_paths:
            logger.warning(
                "No clash-free poses found for %s after %d attempts",
                scaffold_pdb,
                max_attempts,
            )
        return saved_paths

    @staticmethod
    def _heavy_coords(residue):
        return np.asarray([
            atom.coord for atom in residue if atom.element != "H"
        ])

    @staticmethod
    def _has_contact(tree, coords, distance):
        return any(tree.query_ball_point(coords, distance))

    def hotspot_contact_metrics(
        self,
        pdb_path,
        hotspot_residues_str,
        cdr_positions,
        target_chain="A",
        binder_chain="B",
        distance_threshold=5.0,
    ):
        """Return residue-level hotspot and CDR contact metrics."""
        empty = {
            "hotspot_residue_contacts": 0,
            "cdr_hotspot_contacts": 0,
            "cdr_contact_fraction": 0.0,
            "hotspot_contact_fraction": 0.0,
        }
        if not os.path.exists(pdb_path):
            return empty

        structure = self.parser.get_structure("prediction", pdb_path)
        try:
            target = structure[0][target_chain]
            binder = structure[0][binder_chain]
        except KeyError:
            return empty

        target_residues = [
            residue for residue in target if residue.id[0] == " "
        ]
        binder_residues = [
            residue for residue in binder if residue.id[0] == " "
        ]
        if hotspot_residues_str:
            hotspots = set(self._parse_residue_ranges(hotspot_residues_str))
            hotspot_residues = [
                residue for residue in target_residues
                if residue.id[1] in hotspots
            ]
        else:
            hotspot_residues = target_residues
        if not hotspot_residues or not binder_residues:
            return empty

        hotspot_coords = np.concatenate([
            self._heavy_coords(residue) for residue in hotspot_residues
        ])
        binder_coords = np.concatenate([
            self._heavy_coords(residue) for residue in binder_residues
        ])
        hotspot_tree = cKDTree(hotspot_coords)
        binder_tree = cKDTree(binder_coords)

        target_interface = sum(
            self._has_contact(
                binder_tree, self._heavy_coords(residue), distance_threshold
            )
            for residue in target_residues
        )
        hotspot_contacts = sum(
            self._has_contact(
                binder_tree, self._heavy_coords(residue), distance_threshold
            )
            for residue in hotspot_residues
        )
        binder_hotspot_indices = {
            index
            for index, residue in enumerate(binder_residues)
            if self._has_contact(
                hotspot_tree, self._heavy_coords(residue), distance_threshold
            )
        }
        cdr_contacts = len(binder_hotspot_indices & set(cdr_positions or []))
        return {
            "hotspot_residue_contacts": hotspot_contacts,
            "cdr_hotspot_contacts": cdr_contacts,
            "cdr_contact_fraction": (
                cdr_contacts / len(binder_hotspot_indices)
                if binder_hotspot_indices else 0.0
            ),
            "hotspot_contact_fraction": (
                hotspot_contacts / target_interface
                if target_interface else 0.0
            ),
        }
