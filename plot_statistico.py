
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys
 
# ===========================================================================
#                        CONFIGURATION
# ===========================================================================
 
FOCAL_LENGTH    = 550.0   # Camera focal length [px]
CAMERA_OFFSET_Z = 0.16    # Camera is 16 cm below CoM [m]
CAM_W           = 640     # Image width [px]
 
# --- Altitude thresholds ---
# RMSE is computed only below this altitude (= "final approach" phase).
# This excludes the acquisition/chase phase where large errors are expected.
RMSE_ALT_CEILING = 1.5  # [m]  — adjust if needed
 
# Reference altitude for the CEP touchdown measurement.
# We pick the last sample where alt >= this value.
# Must be high enough that pixel→metre conversion is meaningful,
# but low enough to represent the "final landing" accuracy.
CEP_REF_ALT = 0.40  # [m]  — last sample with alt >= 0.40 m
 
# --- File list (edit with your actual CSV filenames) ---
file_list = [
    'log_volo_1775599820.csv',
    'log_volo_1775600020.csv',
    'log_volo_1775600095.csv',
    'log_volo_1775600182.csv',
    'log_volo_1775600230.csv',
    'log_volo_1775600333.csv',
    'log_volo_1775600444.csv',
    'log_volo_1775600528.csv',
    'log_volo_1775600583.csv',
]
 
# ===========================================================================
#                        HELPER FUNCTIONS
# ===========================================================================
 
def extract_errors(df):
    """Extract centred error_x and error_y from a flight log DataFrame."""
    if 'filtered_x' in df.columns:
        error_x = df['filtered_x'] - (CAM_W / 2.0)
    elif 'pos_x_est' in df.columns:
        error_x = df['pos_x_est']  # already centred at 0
    else:
        raise KeyError("Neither 'filtered_x' nor 'pos_x_est' found in CSV")
 
    error_y = df['pos_y_est']
    return error_x, error_y
 
 
def px_to_cm(error_px, alt_m):
    """Convert pixel error to centimetres at a given altitude."""
    z_cam = max(alt_m - CAMERA_OFFSET_Z, 0.05)  # camera-to-ground distance
    return (error_px * z_cam / FOCAL_LENGTH) * 100.0
 
 
# ===========================================================================
#                   1.  ERROR CONVERGENCE PLOT (B&W)
# ===========================================================================
 
fig1, ax1 = plt.subplots(figsize=(10, 6))
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
gray_shades = ['black', 'dimgray', 'darkgray', 'gray', 'silver']
 
for idx, file in enumerate(file_list):
    if not os.path.isfile(file):
        print(f"[WARN] File not found: {file}, skipping.")
        continue
    df = pd.read_csv(file)
    error_x, _ = extract_errors(df)
    sty = line_styles[idx % len(line_styles)]
    col = gray_shades[idx % len(gray_shades)]
    ax1.plot(df['time'], error_x, color=col, linestyle=sty,
             linewidth=2.5, label=f'Trial {idx+1}')
 
ax1.axhline(y=0, color='lightgray', linestyle='-', linewidth=4,
            zorder=0, label=r'Target Setpoint ($e_u = 0$)')
ax1.set_title('Autonomous Alignment: Pixel Error Convergence',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Time [s]', fontsize=12)
ax1.set_ylabel(r'Pixel Error in X-Axis ($e_u$) [px]', fontsize=12)
ax1.set_ylim(-320, 320)
ax1.legend(loc='lower right', fontsize=10, framealpha=1.0, edgecolor='black')
ax1.grid(True, linestyle=':', color='lightgray')
fig1.tight_layout()
fig1.savefig('error_convergence_bw.pdf', format='pdf', dpi=300)
print("Saved: error_convergence_bw.pdf")
 
 
# ===========================================================================
#       2.  CORRECTED STATISTICAL ANALYSIS (per-trial + aggregate)
# ===========================================================================
 
# Per-trial storage
trial_results = []
 
# Aggregate storage for RMSE (final-approach only)
agg_err_x_final = []
agg_err_y_final = []
 
# Aggregate storage for velocity statistics (final-approach only)
agg_vel_x = []
agg_vel_y = []
 
# Touchdown points in cm for scatter plot
td_x_cm_list = []
td_y_cm_list = []
 
for idx, file in enumerate(file_list):
    if not os.path.isfile(file):
        continue
    try:
        df = pd.read_csv(file)
        error_x, error_y = extract_errors(df)
        alt = df['alt']
        vel_x = df['vel_x_cmd']
        vel_y = df['vel_y_cmd']
 
        # -------------------------------------------------------
        # A) RMSE — only during the FINAL APPROACH (alt < ceiling)
        # -------------------------------------------------------
        mask_final = alt < RMSE_ALT_CEILING
        ex_final = error_x[mask_final].dropna().values
        ey_final = error_y[mask_final].dropna().values
        n_final = min(len(ex_final), len(ey_final))
        ex_final = ex_final[:n_final]
        ey_final = ey_final[:n_final]
 
        if n_final > 0:
            rmse_x_px = np.sqrt(np.mean(ex_final**2))
            rmse_y_px = np.sqrt(np.mean(ey_final**2))
            rmse_tot_px = np.sqrt(np.mean(ex_final**2 + ey_final**2))
        else:
            rmse_x_px = rmse_y_px = rmse_tot_px = np.nan
 
        agg_err_x_final.extend(ex_final)
        agg_err_y_final.extend(ey_final)
 
        # Velocity noise in final approach
        vx_final = vel_x[mask_final].dropna().values
        vy_final = vel_y[mask_final].dropna().values
        agg_vel_x.extend(vx_final)
        agg_vel_y.extend(vy_final)
 
        # -------------------------------------------------------
        # B) CEP — at the REFERENCE ALTITUDE
        # -------------------------------------------------------
        # Find the last sample where alt >= CEP_REF_ALT
        valid_idx = df.index[alt >= CEP_REF_ALT]
        if len(valid_idx) > 0:
            li = valid_idx[-1]
            td_ex = error_x.iloc[li]
            td_ey = error_y.iloc[li]
            td_alt = alt.iloc[li]
        else:
            # Fallback: use last available sample
            li = error_x.dropna().index[-1]
            td_ex = error_x.iloc[li]
            td_ey = error_y.iloc[li]
            td_alt = alt.iloc[li]
 
        td_x_cm = px_to_cm(td_ex, td_alt)
        td_y_cm = px_to_cm(td_ey, td_alt)
        td_rad_cm = np.sqrt(td_x_cm**2 + td_y_cm**2)
 
        td_x_cm_list.append(td_x_cm)
        td_y_cm_list.append(td_y_cm)
 
        trial_results.append({
            'trial':      idx + 1,
            'file':       file,
            'n_samples':  n_final,
            'rmse_x_px':  rmse_x_px,
            'rmse_y_px':  rmse_y_px,
            'rmse_tot_px': rmse_tot_px,
            'td_alt_m':   td_alt,
            'td_ex_px':   td_ex,
            'td_ey_px':   td_ey,
            'td_x_cm':    td_x_cm,
            'td_y_cm':    td_y_cm,
            'td_rad_cm':  td_rad_cm,
        })
 
    except Exception as e:
        print(f"Error processing {file}: {e}")
 
# --- Aggregate statistics ---
agg_ex = np.array(agg_err_x_final)
agg_ey = np.array(agg_err_y_final)
 
RMSE_X_PX  = np.sqrt(np.mean(agg_ex**2))  if len(agg_ex) > 0 else np.nan
RMSE_Y_PX  = np.sqrt(np.mean(agg_ey**2))  if len(agg_ey) > 0 else np.nan
RMSE_TOT_PX = np.sqrt(np.mean(agg_ex**2 + agg_ey[:len(agg_ex)]**2)) if len(agg_ex) > 0 else np.nan
 
td_radii = [t['td_rad_cm'] for t in trial_results]
CEP_50_CM = np.median(td_radii) if len(td_radii) > 0 else np.nan
RMSE_X_CM = np.sqrt(np.mean(np.array(td_x_cm_list)**2)) if len(td_x_cm_list) > 0 else np.nan
RMSE_Y_CM = np.sqrt(np.mean(np.array(td_y_cm_list)**2)) if len(td_y_cm_list) > 0 else np.nan
RMSE_TOT_CM = np.sqrt(RMSE_X_CM**2 + RMSE_Y_CM**2)
 
STD_VX = np.std(agg_vel_x) if len(agg_vel_x) > 0 else np.nan
STD_VY = np.std(agg_vel_y) if len(agg_vel_y) > 0 else np.nan
 
 
# ===========================================================================
#                       3.  PRINT RESULTS
# ===========================================================================
 
print("\n" + "=" * 70)
print("           CORRECTED STATISTICAL RESULTS FOR THESIS")
print("=" * 70)
 
print(f"\nConfiguration:")
print(f"  RMSE altitude ceiling:    {RMSE_ALT_CEILING} m")
print(f"  CEP reference altitude:   {CEP_REF_ALT} m")
print(f"  Focal length:             {FOCAL_LENGTH} px")
print(f"  Camera Z offset:          {CAMERA_OFFSET_Z} m")
 
print(f"\n--- PER-TRIAL BREAKDOWN ---")
print(f"{'Trial':>5} | {'RMSE_x':>8} {'RMSE_y':>8} {'RMSE_2D':>8} | "
      f"{'TD alt':>7} {'TD e_x':>7} {'TD e_y':>7} | "
      f"{'X [cm]':>7} {'Y [cm]':>7} {'Rad[cm]':>8}")
print("-" * 95)
for t in trial_results:
    print(f"{t['trial']:>5} | "
          f"{t['rmse_x_px']:>7.1f}px {t['rmse_y_px']:>7.1f}px {t['rmse_tot_px']:>7.1f}px | "
          f"{t['td_alt_m']:>6.2f}m {t['td_ex_px']:>6.1f}px {t['td_ey_px']:>6.1f}px | "
          f"{t['td_x_cm']:>6.2f} {t['td_y_cm']:>6.2f} {t['td_rad_cm']:>7.2f}")
 
print(f"\n--- AGGREGATE RESULTS (N = {len(trial_results)} flights) ---")
print(f"  1. TRACKING RMSE (final approach, alt < {RMSE_ALT_CEILING} m):")
print(f"     RMSE_x  = {RMSE_X_PX:.1f} px")
print(f"     RMSE_y  = {RMSE_Y_PX:.1f} px")
print(f"     RMSE_2D = {RMSE_TOT_PX:.1f} px")
 
print(f"\n  2. TOUCHDOWN ACCURACY (at reference alt ~ {CEP_REF_ALT} m):")
print(f"     RMSE_x  = {RMSE_X_CM:.2f} cm")
print(f"     RMSE_y  = {RMSE_Y_CM:.2f} cm")
print(f"     RMSE_2D = {RMSE_TOT_CM:.2f} cm")
print(f"     CEP_50  = {CEP_50_CM:.2f} cm")
print(f"     (50% of landings within {CEP_50_CM:.2f} cm radial error)")
 
print(f"\n  3. CONTROL EFFORT (final approach):")
print(f"     σ(v_x)  = {STD_VX:.4f} m/s")
print(f"     σ(v_y)  = {STD_VY:.4f} m/s")
print("=" * 70 + "\n")
 
 
# ===========================================================================
#       4.  TOUCHDOWN SCATTER PLOT (thesis-quality figure)
# ===========================================================================
 
if len(td_x_cm_list) >= 2:
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
 
    # --- Left: Scatter plot ---
    ax = axes[0]
    td_x = np.array(td_x_cm_list)
    td_y = np.array(td_y_cm_list)
 
    ax.scatter(td_x, td_y, s=120, c='steelblue', edgecolors='navy',
               linewidths=1.5, zorder=5, label='Touchdown CoM')
    # CEP circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(CEP_50_CM * np.cos(theta), CEP_50_CM * np.sin(theta),
            'purple', linewidth=2, linestyle='--',
            label=f'CEP$_{{50}}$ = {CEP_50_CM:.2f} cm')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, label='Ideal Centre')
 
    lim = max(max(abs(td_x)), max(abs(td_y)), CEP_50_CM) * 1.8 + 1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Lateral Error X [cm]', fontsize=11)
    ax.set_ylabel('Longitudinal Error Y [cm]', fontsize=11)
    ax.set_title('Touchdown Dispersion (CoM)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
 
    # --- Right: Box plot ---
    ax2 = axes[1]
    bp = ax2.boxplot([td_x, td_y], labels=['X (Lateral)', 'Y (Longitudinal)'],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color='red', linewidth=2))
    colors = ['lightsteelblue', 'honeydew']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
 
    ax2.set_ylabel('Error [cm]', fontsize=11)
    ax2.set_title('Statistical Analysis (CoM)', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5, axis='y')
 
    # Annotation box
    stats_text = (f"RMSE$_x$: {RMSE_X_CM:.2f} cm\n"
                  f"RMSE$_y$: {RMSE_Y_CM:.2f} cm\n"
                  f"RMSE$_{{total}}$: {RMSE_TOT_CM:.2f} cm\n"
                  f"CEP$_{{50}}$: {CEP_50_CM:.2f} cm\n"
                  f"σ$_x$: {np.std(td_x):.2f} cm\n"
                  f"σ$_y$: {np.std(td_y):.2f} cm")
    ax2.text(1.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
 
    fig2.tight_layout()
    fig2.savefig('landing_accuracy_analysis.pdf', format='pdf', dpi=300)
    print("Saved: landing_accuracy_analysis.pdf")
 
 
# ===========================================================================
#       5.  ERROR vs ALTITUDE (optical gain explosion visualisation)
# ===========================================================================
 
fig3, ax3 = plt.subplots(figsize=(10, 6))
for idx, file in enumerate(file_list):
    if not os.path.isfile(file):
        continue
    df = pd.read_csv(file)
    error_x, error_y = extract_errors(df)
    alt = df['alt']
    err_magnitude = np.sqrt(error_x**2 + error_y**2)
    col = gray_shades[idx % len(gray_shades)]
    ax3.scatter(alt, err_magnitude, s=3, alpha=0.4, color=col, label=f'Trial {idx+1}')
 
# Overlay the dynamic alignment threshold
alt_range = np.linspace(0.2, 5.0, 200)
cone = np.interp(alt_range, [0.7, 2.0, 5.0], [1.0, 1.25, 1.5])
threshold = 110.0 * cone
ax3.plot(alt_range, threshold, 'r--', linewidth=2,
         label=r'Dynamic threshold $\varepsilon(h)$')
 
ax3.set_xlabel('Altitude [m]', fontsize=12)
ax3.set_ylabel('2D Pixel Error Magnitude [px]', fontsize=12)
ax3.set_title('Tracking Error vs. Altitude', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.set_xlim(0, 5.5)
fig3.tight_layout()
fig3.savefig('error_vs_altitude.pdf', format='pdf', dpi=300)
print("Saved: error_vs_altitude.pdf")
 
 
# ===========================================================================
#       6.  VELOCITY COMMAND HISTOGRAM
# ===========================================================================
 
if len(agg_vel_x) > 0:
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
 
    ax4a.hist(agg_vel_x, bins=50, color='steelblue', edgecolor='navy', alpha=0.8)
    ax4a.axvline(0, color='red', linewidth=1, linestyle='--')
    ax4a.set_xlabel('$v_x$ (Pitch) [m/s]', fontsize=11)
    ax4a.set_ylabel('Count', fontsize=11)
    ax4a.set_title(f'Pitch Velocity Commands (σ={STD_VX:.4f} m/s)',
                   fontsize=12, fontweight='bold')
    ax4a.grid(True, linestyle=':', alpha=0.5)
 
    ax4b.hist(agg_vel_y, bins=50, color='darkorange', edgecolor='saddlebrown', alpha=0.8)
    ax4b.axvline(0, color='red', linewidth=1, linestyle='--')
    ax4b.set_xlabel('$v_y$ (Roll) [m/s]', fontsize=11)
    ax4b.set_ylabel('Count', fontsize=11)
    ax4b.set_title(f'Roll Velocity Commands (σ={STD_VY:.4f} m/s)',
                   fontsize=12, fontweight='bold')
    ax4b.grid(True, linestyle=':', alpha=0.5)
 
    fig4.suptitle('Control Effort Distribution (Final Approach)', fontsize=13, fontweight='bold')
    fig4.tight_layout()
    fig4.savefig('velocity_histogram.pdf', format='pdf', dpi=300)
    print("Saved: velocity_histogram.pdf")
 
plt.show()
print("\nDone.")