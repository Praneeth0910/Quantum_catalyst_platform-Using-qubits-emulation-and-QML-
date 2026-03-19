import py3Dmol
import streamlit.components.v1 as components

def mol_to_xyz(mol):
    """
    Converts RDKit molecule to XYZ format string.
    """
    conf = mol.GetConformer()
    atoms = mol.GetAtoms()
    xyz = f"{len(atoms)}\n\n"
    for atom in atoms:
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz += f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"
    return xyz

def show_molecule_3d(mol):
    """
    Returns a Py3Dmol viewer for the molecule.
    """
    xyz = mol_to_xyz(mol)
    viewer = py3Dmol.view(width=500, height=400)
    viewer.addModel(xyz, "xyz")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    html = viewer._make_html()
    components.html(html, height=400)
