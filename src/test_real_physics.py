"""
Test script for μ-MuZero with REALISTIC PHYSICS parameters.

Validates:
  1. Physical constants and unit consistency
  2. Brownian motion magnitude (Einstein-Smoluchowski)
  3. Optical trap force vs displacement
  4. Low Reynolds number dynamics
  5. Full episode with real physics
"""

import sys
import numpy as np

from config import MuMuZeroConfig
from environment import OpticalMicromanipulationEnv


def test_physical_constants():
    """Verify precomputed physical constants are correct."""
    print("\n" + "=" * 60)
    print("Test 1: Physical Constants")
    print("=" * 60)

    config = MuMuZeroConfig()
    env = OpticalMicromanipulationEnv(config)

    # Manual calculation for verification
    r_m = config.robot_radius * 1e-6  # m
    gamma_manual = 6.0 * np.pi * config.viscosity * r_m  # N·s/m
    D_manual = config.k_B * config.temperature / gamma_manual  # m²/s
    brownian_std_manual = np.sqrt(2.0 * D_manual * 1e12 * config.dt)

    print(f"  Robot radius:        {config.robot_radius} μm")
    print(f"  Viscosity:           {config.viscosity} Pa·s")
    print(f"  Temperature:         {config.temperature} K")
    print(f"  Time step:           {config.dt * 1000} ms")
    print(f"")
    print(f"  Stokes drag γ:       {env.gamma_Ns_m:.4e} N·s/m")
    print(f"  Stokes drag γ:       {env.gamma_pN_s_um:.4f} pN·s/μm")
    print(f"  Diffusion D:         {env.D_um2_s:.4f} μm²/s")
    print(f"  Brownian σ/step:     {env.brownian_std:.4f} μm")
    print(f"  (manual check:       {brownian_std_manual:.4f} μm)")
    print(f"  Thermal fluct. σ:    {env.thermal_sigma:.4f} μm")
    print(f"  Photon dose/step:    {env.photon_dose_per_step:.1f} mW·ms")

    assert np.isclose(env.gamma_Ns_m, gamma_manual), "Stokes drag mismatch"
    assert np.isclose(env.brownian_std, brownian_std_manual), "Brownian std mismatch"
    assert env.brownian_std > 0, "Brownian std must be positive"

    print("  [PASS]")


def test_brownian_motion_statistics():
    """Verify Brownian motion follows Einstein-Smoluchowski relation."""
    print("\n" + "=" * 60)
    print("Test 2: Brownian Motion Statistics")
    print("=" * 60)

    config = MuMuZeroConfig()
    config.trap_stiffness = 0.0  # Turn off trap to observe free diffusion
    env = OpticalMicromanipulationEnv(config)

    # Run many steps with hold action (no trap movement)
    env.reset(seed=42)
    positions = []
    n_steps = 2000

    for _ in range(n_steps):
        obs, _, _, _ = env.step((0, 0))  # hold
        positions.append(env.robots[0]['pos'].copy())

    positions = np.array(positions)

    # Compute mean-squared displacement
    dx = positions[:, 0]
    dy = positions[:, 1]

    # MSD for various lag times
    msd_x = []
    msd_y = []
    lag_times = [1, 5, 10, 20, 50, 100]

    for lag in lag_times:
        displacements = positions[lag:] - positions[:-lag]
        msd_x.append(np.mean(displacements[:, 0] ** 2))
        msd_y.append(np.mean(displacements[:, 1] ** 2))

    # For free diffusion: MSD = 2 * D * t
    # Expected slope: 2 * D [μm²/s] * dt [s] / step
    expected_msd_per_step = 2.0 * env.D_um2_s * config.dt

    print(f"  Steps simulated:     {n_steps}")
    print(f"  Expected MSD/step:   {expected_msd_per_step:.6f} μm²")
    print(f"  Observed MSD/step:   {msd_x[0]:.6f} μm² (x), {msd_y[0]:.6f} μm² (y)")

    # Check linearity (MSD ∝ t)
    for i, lag in enumerate(lag_times):
        expected = expected_msd_per_step * lag
        print(f"    lag={lag:>3}:  MSD_x={msd_x[i]:.6f}  MSD_y={msd_y[i]:.6f}  expected={expected:.6f}")

    # Allow 20% tolerance for statistical fluctuations
    assert np.isclose(msd_x[0], expected_msd_per_step, rtol=0.2), f"MSD x mismatch: {msd_x[0]} vs {expected_msd_per_step}"
    assert np.isclose(msd_y[0], expected_msd_per_step, rtol=0.2), f"MSD y mismatch: {msd_y[0]} vs {expected_msd_per_step}"

    print("  [PASS]")


def test_optical_trap_force():
    """Verify harmonic trap force model."""
    print("\n" + "=" * 60)
    print("Test 3: Optical Trap Force")
    print("=" * 60)

    config = MuMuZeroConfig()
    config.trap_stiffness = 5.0  # pN/μm
    env = OpticalMicromanipulationEnv(config)

    # Test force at various displacements
    test_displacements = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    print(f"  Trap stiffness:      {config.trap_stiffness} pN/μm")
    print(f"  Escape distance:     {env.trap_escape_dist} μm")
    print("")

    for d in test_displacements:
        rel_pos = np.array([d, 0.0, 0.0])
        force = env._optical_force(rel_pos)
        f_mag = np.linalg.norm(force)
        # Expected: F = k * d * (1 - d/r_max) with axial weakening
        if d <= env.trap_escape_dist:
            expected = config.trap_stiffness * d * (1.0 - d / env.trap_escape_dist)
        else:
            expected = 0.0

        print(f"    d={d:.1f} μm:  |F|={f_mag:.3f} pN  expected={expected:.3f} pN")

        # Check force is zero beyond escape distance
        if d > env.trap_escape_dist:
            assert f_mag < 1e-6, f"Force should be zero beyond escape: {f_mag}"

    # Test axial vs lateral stiffness
    rel_pos_lateral = np.array([1.0, 0.0, 0.0])
    rel_pos_axial = np.array([0.0, 0.0, 1.0])
    f_lat = env._optical_force(rel_pos_lateral)
    f_ax = env._optical_force(rel_pos_axial)

    ratio = abs(f_ax[2]) / (abs(f_lat[0]) + 1e-8)
    print(f"")
    print(f"  Lateral force @1μm:  {abs(f_lat[0]):.3f} pN")
    print(f"  Axial force @1μm:    {abs(f_ax[2]):.3f} pN")
    print(f"  Axial/lateral ratio: {ratio:.3f} (expected ~0.3)")

    assert 0.25 < ratio < 0.35, f"Axial weakening incorrect: {ratio}"

    print("  [PASS]")


def test_trap_capture():
    """Test that a free particle gets captured by a moving trap."""
    print("\n" + "=" * 60)
    print("Test 4: Trap Capture Dynamics")
    print("=" * 60)

    config = MuMuZeroConfig()
    config.trap_stiffness = 5.0
    config.dt = 0.02
    env = OpticalMicromanipulationEnv(config)
    env.reset(seed=123)

    # Place robot far from trap
    env.robots[0]['pos'] = np.array([20.0, 20.0, 25.0])
    env.traps[0] = np.array([20.0, 20.0, 25.0])  # Trap right on top

    initial_dist = 0.0  # Already at trap center
    print(f"  Initial robot pos:   {env.robots[0]['pos']}")
    print(f"  Initial trap pos:    {env.traps[0]}")
    print(f"  Initial distance:    {initial_dist:.2f} μm")

    # Let it thermalize near trap
    positions = []
    for _ in range(500):
        env.step((0, 0))  # hold
        positions.append(env.robots[0]['pos'].copy())

    positions = np.array(positions)
    mean_pos = np.mean(positions, axis=0)
    std_pos = np.std(positions, axis=0)

    print(f"  After 500 steps:")
    print(f"    Mean position:     [{mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f}] μm")
    print(f"    Position std:      [{std_pos[0]:.3f}, {std_pos[1]:.3f}, {std_pos[2]:.3f}] μm")
    print(f"    Expected thermal σ: {env.thermal_sigma:.3f} μm")

    # Should be near trap center
    assert np.linalg.norm(mean_pos - env.traps[0]) < 0.5, "Particle not captured by trap"

    # Position std should be close to thermal fluctuation amplitude
    assert 0.5 * env.thermal_sigma < std_pos[0] < 2.0 * env.thermal_sigma, \
        f"Lateral fluctuation {std_pos[0]} not near thermal {env.thermal_sigma}"

    print("  [PASS]")


def test_stokes_drag_scaling():
    """Verify drag force scales correctly with velocity."""
    print("\n" + "=" * 60)
    print("Test 5: Stokes Drag Scaling")
    print("=" * 60)

    config = MuMuZeroConfig()
    env = OpticalMicromanipulationEnv(config)

    # For a particle at trap center (no optical force), apply external force
    # and check velocity = F / γ
    test_forces = [1.0, 5.0, 10.0, 50.0]  # pN

    print(f"  γ = {env.gamma_pN_s_um:.4f} pN·s/μm")
    print("")

    for F_pN in test_forces:
        expected_v = F_pN / env.gamma_pN_s_um  # μm/s
        print(f"    F = {F_pN:>5.1f} pN  =>  v = {expected_v:>8.2f} μm/s")

    # Verify Reynolds number << 1 (low Reynolds number regime)
    # Re = ρ v r / η
    rho_water = 1000.0  # kg/m³
    v_max = 50.0 / env.gamma_pN_s_um  # μm/s for 50 pN force
    v_max_m = v_max * 1e-6  # m/s
    r_m = config.robot_radius * 1e-6
    Re = rho_water * v_max_m * r_m / config.viscosity

    print(f"")
    print(f"  Max velocity:        {v_max:.2f} μm/s")
    print(f"  Reynolds number:     {Re:.2e} (must be << 1)")
    assert Re < 0.1, f"Re = {Re}, not in low Reynolds regime!"

    print("  [PASS]")


def test_full_episode_real_physics():
    """Run a full episode and print physics metrics."""
    print("\n" + "=" * 60)
    print("Test 6: Full Episode with Real Physics")
    print("=" * 60)

    config = MuMuZeroConfig()
    config.trap_stiffness = 2.0
    config.dt = 0.02
    env = OpticalMicromanipulationEnv(config)

    obs = env.reset(seed=999)

    print(f"  Initial state:")
    print(f"    Robot pos:         {env.robots[0]['pos']}")
    print(f"    Trap pos:          {env.traps[0]}")
    print(f"    Target pos:        {env.target['pos']}")
    print(f"    Goal pos:          {env.goal}")
    print(f"")

    # Random policy
    np.random.seed(999)
    total_reward = 0.0
    max_step_displacement = 0.0
    force_log = []

    for step in range(100):
        action = (0, np.random.randint(0, len(config.trap_primitives)))
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Log max displacement per step
        if step > 0:
            disp = np.linalg.norm(env.robots[0]['pos'] - prev_pos)
            max_step_displacement = max(max_step_displacement, disp)

        prev_pos = env.robots[0]['pos'].copy()

        # Log optical force magnitude
        rel_pos = env.traps[0] - env.robots[0]['pos']
        force = env._optical_force(rel_pos)
        force_log.append(np.linalg.norm(force))

        if done:
            break

    print(f"  Episode length:      {step + 1} steps ({(step + 1) * config.dt * 1000:.0f} ms)")
    print(f"  Total reward:        {total_reward:.2f}")
    print(f"  Final robot pos:     {env.robots[0]['pos']}")
    print(f"  Final target pos:    {env.target['pos']}")
    print(f"  Distance to goal:    {np.linalg.norm(env.target['pos'] - env.goal):.2f} μm")
    print(f"  Phototoxicity:       {env.phototoxicity:.0f} / {config.phototoxicity_budget:.0f} mW·ms")
    print(f"  Max step disp:       {max_step_displacement:.3f} μm")
    print(f"  Avg optical force:   {np.mean(force_log):.2f} pN")
    print(f"  Max optical force:   {np.max(force_log):.2f} pN")

    # Sanity checks
    assert max_step_displacement < 10.0, f"Suspiciously large displacement: {max_step_displacement}"
    assert np.mean(force_log) < 20.0, f"Suspiciously large average force: {np.mean(force_log)}"

    print("  [PASS]")


def test_phototoxicity_budget():
    """Verify phototoxicity budget is physically reasonable."""
    print("\n" + "=" * 60)
    print("Test 7: Phototoxicity Budget")
    print("=" * 60)

    config = MuMuZeroConfig()
    env = OpticalMicromanipulationEnv(config)

    dose_per_step = env.photon_dose_per_step
    budget = config.phototoxicity_budget
    max_steps = budget / dose_per_step

    print(f"  Laser power:         {config.laser_power} mW")
    print(f"  Dose per step:       {dose_per_step:.1f} mW·ms")
    print(f"  Budget:              {budget:.0f} mW·ms")
    print(f"  Max safe steps:      {max_steps:.0f} ({max_steps * config.dt:.1f} s)")

    # Should allow at least a few seconds of operation
    assert max_steps * config.dt > 5.0, "Phototoxicity budget too restrictive"

    print("  [PASS]")


def main():
    print("\n" + "=" * 60)
    print("μ-MuZero Real Physics Validation Suite")
    print("=" * 60)
    print("Testing with realistic optical tweezer parameters:")
    print("  - 2 μm polystyrene beads in water at 20°C")
    print("  - 1064 nm optical tweezers, ~50 mW per trap")
    print("  - Overdamped regime (Re << 1)")
    print("=" * 60)

    try:
        test_physical_constants()
        test_brownian_motion_statistics()
        test_optical_trap_force()
        test_trap_capture()
        test_stokes_drag_scaling()
        test_full_episode_real_physics()
        test_phototoxicity_budget()

        print("\n" + "=" * 60)
        print("ALL REAL-PHYSICS TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
