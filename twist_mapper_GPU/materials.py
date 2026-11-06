"""
materials.py — Crystal structure definitions for 2D materials.

Provides a Material dataclass and a registry of common 2D materials
used in HAADF-STEM twist angle analysis. Ported over from the earlier
twist_mapper package's twist_mapper/preprocessing/materials.py (see the
vectorized branch) — content unchanged except for the doctring's usage
example below, updated for this package's own API (TileFFTAnalysis takes
pixel_size_ang/d_spacing_ang directly; there is no ImageStack here).

Design
------
Material is a @dataclass with three user-settable fields:

    Settable fields (managed by @dataclass __init__):
        name                 str    — human-readable material name
        lattice_constant_ang float  — in-plane lattice constant a (Å)
        lattice_type         str    — 'hexagonal' (default) or 'square'

    Class methods:
        Material.get(name)   — look up from registry (case-insensitive)

    Note: d_spacing_ang and expected_ring_radius are NOT derived here.
    Material knows only universal crystal constants. d_spacing depends
    on which plane is imaged (an experimental choice); expected_ring_radius
    additionally depends on pixel size and tile size (properties of the
    acquisition/tiling, not the crystal) — see
    TileFFTAnalysis.expected_ring_radius(tile_size, pixel_size_ang,
    d_spacing_ang).

Usage
-----
    from twist_mapper_GPU.materials import Material, MATERIALS, get_material
    from twist_mapper_GPU import TileFFTAnalysis

    mat = get_material('WSSe')          # or Material.get('WSSe')
    print(mat.lattice_constant_ang)     # 3.25 Å
    print(mat.lattice_type)             # 'hexagonal'

    # Modify after construction
    mat.lattice_constant_ang = 3.27

    # Add a custom material
    MATERIALS['MyAlloy'] = Material('MyAlloy', lattice_constant_ang=3.20)

    # d_spacing_ang for a given plane is still an experimental choice, not
    # derived from Material -- pass it (and pixel_size_ang) straight to
    # expected_ring_radius, e.g. using the (1,0) plane spacing a*sqrt(3)/2
    # for a hexagonal lattice:
    d_spacing_ang = mat.lattice_constant_ang * (3 ** 0.5) / 2
    expected_r = TileFFTAnalysis.expected_ring_radius(
        tile_size=128, pixel_size_ang=0.25, d_spacing_ang=d_spacing_ang
    )
"""

from dataclasses import dataclass


# ------------------------------------------------------------------
# Material dataclass
# ------------------------------------------------------------------

@dataclass
class Material:
    """Crystal structure definition for a 2D material.

    Parameters
    ----------
    name : str
        Human-readable material name (e.g. 'WSSe', 'MoS2').
    lattice_constant_ang : float
        In-plane lattice constant a in Angstroms.  Must be > 0.
    lattice_type : str
        'hexagonal' (default) or 'square'.

    Notes
    -----
    @dataclass generates __init__, __repr__, and __eq__ automatically
    from the three annotated fields.

    d_spacing_ang and expected_ring_radius are NOT on Material — they
    depend on experimental choices (which plane is imaged, pixel size,
    tile size), not just the crystal structure. See
    TileFFTAnalysis.expected_ring_radius.
    """

    name: str
    lattice_constant_ang: float
    lattice_type: str = 'hexagonal'

    def __post_init__(self):
        """Validate fields after dataclass-generated __init__ runs."""
        if self.lattice_type not in ('hexagonal', 'square'):
            raise ValueError(
                f"lattice_type must be 'hexagonal' or 'square', "
                f"got '{self.lattice_type}'. "
            )
        if self.lattice_constant_ang <= 0:
            raise ValueError(
                f"lattice_constant_ang must be > 0, "
                f"got {self.lattice_constant_ang}."
            )

    # ------------------------------------------------------------------
    # Class method — registry lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> 'Material':
        """Look up a material from the registry by name (case-insensitive).

        Convenience alias for get_material(name).

        Parameters
        ----------
        name : str
            Material name, e.g. 'WSSe', 'MoS2', 'graphene'.

        Returns
        -------
        Material

        Raises
        ------
        KeyError
            If the material is not found in the registry.

        Examples
        --------
        >>> mat = Material.get('WSSe')
        >>> mat.lattice_constant_ang
        3.25
        """
        return get_material(name)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Material('{self.name}', "
            f"a={self.lattice_constant_ang} Å, "
            f"{self.lattice_type})"
        )


# ------------------------------------------------------------------
# Registry of common 2D materials
# ------------------------------------------------------------------

MATERIALS = {
    # Graphene family
    'graphene' : Material('graphene', lattice_constant_ang=2.46),
    'hBN'      : Material('hBN',      lattice_constant_ang=2.50),

    # Molybdenum dichalcogenides
    'MoS2'     : Material('MoS2',     lattice_constant_ang=3.16),
    'MoSe2'    : Material('MoSe2',    lattice_constant_ang=3.29),
    'MoTe2'    : Material('MoTe2',    lattice_constant_ang=3.52),

    # Tungsten dichalcogenides
    'WS2'      : Material('WS2',      lattice_constant_ang=3.18),
    'WSe2'     : Material('WSe2',     lattice_constant_ang=3.28),
    'WSSe'     : Material('WSSe',     lattice_constant_ang=3.25),
}


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def get_material(name: str) -> Material:
    """Look up a material by name from the registry (case-insensitive).

    Parameters
    ----------
    name : str
        Material name, e.g. 'WSSe', 'MoS2', 'graphene'.

    Returns
    -------
    Material

    Raises
    ------
    KeyError
        If the material is not found.  Available names are listed
        in the error message.

    Examples
    --------
    >>> mat = get_material('wsse')   # case-insensitive
    >>> mat.name
    'WSSe'
    """
    if name in MATERIALS:
        return MATERIALS[name]

    # case-insensitive fallback
    name_lower = name.lower()
    for key, mat in MATERIALS.items():
        if key.lower() == name_lower:
            return mat

    available = ', '.join(sorted(MATERIALS.keys()))
    raise KeyError(
        f"Material '{name}' not found in registry. "
        f"Available: {available}. "
        f"Add custom materials via: "
        f"MATERIALS['MyMaterial'] = Material('MyMaterial', "
        f"lattice_constant_ang=3.10)"
    )
