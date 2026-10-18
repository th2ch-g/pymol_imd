# Scripted molecular dragging

Use this variant only when the user requests code-driven dragging. Use
`pymol_imd.scripted.run_drag`, not direct position assignment or an additional
bond-restraint force. These are externally steered demonstrations, recorded
separately from computer-use trials. The general API and setup example are in
`README.md` under "Scripted dragging without a viewer".

Use `nsteps_per_frame=1` and the backend's recommended timestep. Minimize, save
a checkpoint, then compare the no-force control and steering run from that
same checkpoint, including velocities and random-generator state. Use a fresh
output directory for each run. Save a prepared initial PDB matching the XTC
atom order. After steering, use `release_ps` to continue with zero pull force.
Do not regenerate a numerical-regression reference to make these tests pass.

A callback receives time in ps and full-system positions in angstrom. It must
return virtual cursor offsets for the selected atoms in their selected order.
For a desired force array in kJ/mol/nm, convert it with
`offsets = forces / system.config.const_k * constants.NM_TO_ANGSTROM`.
The normal drag noise threshold and any explicitly configured force cap apply.
To bound a target force smoothly, use
`forces = raw / sqrt(1 + (norm(raw, axis=1) / soft_cap)**2)[:, None]`.
Record `soft_cap` separately from the unchanged system force cap.

## QBP domain opening

Use `examples/qbp.pdb` and the open reference. Align reference domain A
(residues 5-84 and 186-224) to the initial domain A using common CA atoms
(AA) or BB-to-CA correspondences (CG). Fit the initial domain B (90-180) to
the aligned reference B to obtain a rigid rotation and translation. Interpolate
that rigid transform over 20 ps, hold the final target for 5 ps, then release
for 10 ps. Interpolate the rotation on its axis, rather than interpolating
coordinates through a deformation of the domain.

At each step, fit initial domain A to current domain A and move the target
with that global transform. Build forces from the target-minus-current
positions in nm. A tested AA protocol uses domain B only, target stiffness
1000 kJ/mol/nm^2 and per-atom soft cap 250 kJ/mol/nm. AA keeps Amber/GBn2,
300 K and a 0.002 ps timestep.

For CG, select both domains. Move B toward its interpolated rigid target and
A toward its initial internal shape under the current best-fit A transform.
The additional A selection limits its deformation while B is pulled. A tested
protocol uses stiffness 10000 kJ/mol/nm^2 and soft cap 2000 kJ/mol/nm. Keep
Martini 3's default elastic network, automatic periodic box, 300 K and 0.02 ps
timestep; do not silently change those conditions. This is an explicit force
on both selected domains, not a B-only drag.

Analyze the open and closed reference RMSDs, domain-internal RMSDs and centroid
distance over the entire trajectory, especially at the end of the hold and
release. CG BB-to-reference-CA RMSD is approximate. CG bead centroid distance
is not the paper's heavy-atom observable. Opening under applied force and
remaining open after release are separate outcomes. Report a return toward
closed after release even if the driven geometry matched the open reference.

## Chorismate to prephenate

Use the fixture and ML/MM selection in `chorismate.md`: AIMNet2, region charge
0, multiplicity 1, 160 selected atoms and 12 link hydrogens. Keep the default
GBn2 environment and the fixture's single bridging water. Use a 0.0005 ps
timestep, 300 K, the normal force scale 50000 kJ/mol/nm^2 and an explicitly configured force cap of
5000 kJ/mol/nm for this scripted protocol. Record the actual devices and seed.

Resolve the four ligand atoms by name in the prepared topology. Select them
in pair order: C3, C16, C1, O11. Measure the initial pair distances after
minimization. Interpolate targets from those distances to `(2.81, 2.19)`
angstrom at 2 ps, then to `(1.56, 2.95)` at 4 ps. Hold to 4.5 ps, then run an
unforced 1 ps release segment. These numbers are steering targets, not
measured transition-state coordinates or evidence of a transition state.

For each pair, compute `error_nm = (distance_A - target_A) * 0.1` and
`f = -const_k * error_nm / sqrt(1 + (const_k * error_nm / force_cap)**2)`.
Apply `f` along the unit vector from the second atom to the first, with an
equal opposite force on the second atom. Convert those forces to cursor
offsets before calling the API. The coordinates remain controlled by MD.

Check all six bond changes listed in `chorismate.md`, then independently infer
ligand connectivity and stereochemistry from recorded coordinates with RDKit.
Check every release frame, not just the most favorable frame. Save a product
SDF with inferred bonds if assignment succeeds; the full-system PDB and XTC
retain the initial topology. Report any force clipping or integration failure.
A coordinate-inferred prephenate structure supports externally driven
conversion in this model; it does not establish a concerted transition state,
activation barrier or spontaneous enzymatic catalysis. Label an unmatched or
omitted control explicitly.
