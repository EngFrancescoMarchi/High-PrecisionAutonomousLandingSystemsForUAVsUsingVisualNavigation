import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_full_landing_analysis(csvfilepath):
    print(">>> GENERATING FULL LANDING ANALYSIS... <<<")
    
    # 1. System Parameters
    cm_per_pixel = 0.12
    camera_offset_y_px = 110  # Camera to CoM distance in pixels
    
    # Load data
    try:
        df = pd.read_csv('/home/marchi/High-PrecisionAutonomousLandingSystemsForUAVsUsingVisualNavigation/Graphs/dati_atterraggio.csv')
    except FileNotFoundError:
        print(f"Error: File '{'/home/marchi/High-PrecisionAutonomousLandingSystemsForUAVsUsingVisualNavigation/Graphs/dati_atterraggio.csv'}' not found.")
        return

    # 2. Error Calculations in cm
    df['Err X cm'] = df['Err X pixel'] * cm_per_pixel
    
    # Camera Error (With Offset)
    df['Camera Err Y cm'] = df['ErrY pixel'] * cm_per_pixel
    
    # Center of Mass Error (Without Offset)
    df['CoM Err Y cm'] = (df['ErrY pixel'] - camera_offset_y_px) * cm_per_pixel
    
    # Distance from center for CoM
    df['Distance cm'] = np.sqrt(df['Err X cm']**2 + df['CoM Err Y cm']**2)
    
    # --- STATISTICAL CALCULATIONS (Based on CoM) ---
    rmse_x = np.sqrt(np.mean(df['Err X cm']**2))
    rmse_y = np.sqrt(np.mean(df['CoM Err Y cm']**2))
    rmse_tot = np.sqrt(np.mean(df['Distance cm']**2))
    cep_50 = np.median(df['Distance cm']) # 50% Circular Error Probable
    std_x = np.std(df['Err X cm'])
    std_y = np.std(df['CoM Err Y cm'])

    # 3. Create Figure with 3 Subplots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    
    # Function to draw the physical target background
    def draw_target_background(ax, title):
        # Physical Base (60x60 cm)
        dim_base = 60
        ax.add_patch(patches.Rectangle((-dim_base/2, -dim_base/2), dim_base, dim_base, 
                                       linewidth=1, edgecolor='gray', facecolor='silver', 
                                       alpha=0.6, zorder=0, label='Physical Base (60x60)'))
        # Visual Target (55x55 cm)
        dim_target = 55
        ax.add_patch(patches.Rectangle((-dim_target/2, -dim_target/2), dim_target, dim_target, 
                                         linewidth=2, edgecolor='black', facecolor='white', 
                                         zorder=1, label='Visual Target (55x55)'))
        # Ideal Center
        ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=3, zorder=11, label='Ideal Center (0,0)')
        
        # Axis formatting
        ax.axhline(0, color='black', linewidth=1, zorder=5)
        ax.axvline(0, color='black', linewidth=1, zorder=5)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_xlabel('Lateral Error X [cm]', fontsize=12)
        ax.set_ylabel('Longitudinal Error Y [cm]', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.5, zorder=2)

    # ==============================================================================
    # PANEL 1 (Left): CAMERA SENSOR POSITION (With Offset)
    # ==============================================================================
    draw_target_background(ax1, 'Camera Sensor Position\n(Hardware Offset Present)')
    
    ax1.scatter(df['Err X cm'], df['Camera Err Y cm'], color='orange', alpha=0.9, 
                edgecolors='black', s=100, zorder=10, label='Camera Coordinates')
    ax1.legend(loc='upper right', fontsize=9)

    # ==============================================================================
    # PANEL 2 (Middle): DRONE CENTER OF MASS (Without Offset)
    # ==============================================================================
    draw_target_background(ax2, 'Drone Center of Mass (CoM)\n(Actual Landing Position)')
    
    ax2.scatter(df['Err X cm'], df['CoM Err Y cm'], color='dodgerblue', alpha=0.9, 
                edgecolors='black', s=100, zorder=10, label='Drone Body (CoM)')
    
    # Draw CEP50 Circle
    cep_circle = plt.Circle((0, 0), cep_50, color='purple', fill=False, linestyle='--', 
                            linewidth=2, zorder=5, label=f'CEP50 ({cep_50:.1f} cm)')
    ax2.add_patch(cep_circle)
    ax2.legend(loc='upper right', fontsize=9)

    # ==============================================================================
    # PANEL 3 (Right): BOXPLOTS & STATS (Based on CoM)
    # ==============================================================================
    box_data = [df['Err X cm'], df['CoM Err Y cm']]
    bp = ax3.boxplot(box_data, patch_artist=True, labels=['X Axis\n(Lateral)', 'Y Axis\n(Longitudinal)'])
    
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Error [cm]', fontsize=12)
    ax3.set_title('Statistical Analysis (CoM)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)

    # Text box with precision metrics
    stats_text = (
        "$\mathbf{Precision\ Metrics:}$\n\n"
        f"$\mathbf{{RMSE_x}}:$ {rmse_x:.2f} cm\n"
        f"$\mathbf{{RMSE_y}}:$ {rmse_y:.2f} cm\n"
        f"$\mathbf{{RMSE_{{total}}}}:$ {rmse_tot:.2f} cm\n\n"
        f"$\mathbf{{CEP_{{50}}}}:$ {cep_50:.2f} cm\n"
        "(50% of landings fall\ninside this radius)\n\n"
        f"$\mathbf{{\sigma_x}}$ (Std Dev): {std_x:.2f} cm\n"
        f"$\mathbf{{\sigma_y}}$ (Std Dev): {std_y:.2f} cm"
    )
    
    props = dict(boxstyle='round,pad=1', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
    ax3.text(1.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='center', bbox=props)

    # Adjust layout to fit everything including the text box on the right
    plt.subplots_adjust(right=0.85, wspace=0.3)
    
    # Save the plot
    output_name = 'landing_full_analysis.png'
    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    print(f">>> Plot successfully saved as '{output_name}' <<<")
    plt.close(fig)

# CALL THE FUNCTION
plot_full_landing_analysis('dati_atterraggio.csv')