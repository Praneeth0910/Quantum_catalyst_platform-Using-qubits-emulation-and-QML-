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

def show_molecule_3d(mol, width: int = 600, height: int = 450, style: str = "stick"):
    """
    Renders an interactive, draggable 3D model of the molecule using py3Dmol.

    Args:
        mol: RDKit molecule with a 3D conformer.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
        style: Rendering style — "stick", "sphere", or "ballstick".
    """
    xyz = mol_to_xyz(mol)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(xyz, "xyz")

    if style == "sphere":
        viewer.setStyle({"sphere": {"scale": 0.3, "colorscheme": "Jmol"}})
    elif style == "ballstick":
        viewer.setStyle({"stick": {"radius": 0.15, "colorscheme": "Jmol"}, "sphere": {"scale": 0.25, "colorscheme": "Jmol"}})
    else:
        viewer.setStyle({"stick": {"radius": 0.15, "colorscheme": "Jmol"}})

    viewer.setBackgroundColor("#0e1117")  # Match Streamlit dark background
    viewer.zoomTo()
    viewer.zoom(0.9)

    html = viewer._make_html()
    components.html(html, height=height, scrolling=False)
