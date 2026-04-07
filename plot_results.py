from turtle import pd

import matplotlib.pyplot as plt
import numpy as np
def plot_results(data):
    print(">>> GENERATING GRAPHS IN PROGRESS... <<<")
    
    # Create a figure with 3 stacked plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Altitude
    ax1.plot(data['time'], data['alt'], 'b-', label='Altitude')
    ax1.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('Mission Profile')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Position Error (Kalman Estimate)
    ax2.plot(data['time'], data['pos_x_est'], 'r-', label='Error X (Est)')
    ax2.plot(data['time'], data['pos_y_est'], 'g-', label='Error Y (Est)')
    ax2.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
    ax2.set_ylabel('Position Error [Pixels]') 
    ax2.set_title('Tracking Error (Kalman Estimate)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Velocity Commands
    ax3.plot(data['time'], data['vel_x_cmd'], 'r--', label='Cmd Vel X')
    ax3.plot(data['time'], data['vel_y_cmd'], 'g--', label='Cmd Vel Y')
    
    # Add target visibility indicator (gray area when lost)
    # Scale to make it visible
    vis = np.array(data['target_visible'])
    # Draw red background where target is lost
    # (A bit complex logic for fast matplotlib, let's keep it simple:)
    # ax3.fill_between(data['time'], -1, 1, where=(vis==0), color='red', alpha=0.1, label='Target Lost')

    ax3.set_ylabel('Velocity Cmd [m/s]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Control Outputs')
    ax3.grid(True)
    ax3.legend()
    
    # Save and Show
    plt.savefig('mission_log.png')
    print(">>> Graph saved as 'mission_log.png' <<<")
    #plt.show()
def plot_tuning_graph(log_data_dict):
    df = pd.DataFrame(log_data_dict)
        
    # Crea la figura con stile accademico
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Setpoint (Linea tratteggiata rossa)
    plt.plot(df['time'], df['setpoint_x'], 'r--', linewidth=2, label='Setpoint (Image Center)')
    
    # 2. Plot Misura Grezza (Linea sottile semi-trasparente)
    plt.plot(df['time'], df['raw_x'], color='lightgray', linewidth=1.5, alpha=0.8, label='Raw ArUco Detection')
    
    # 3. Plot Misura Filtrata / Risposta (Linea spessa blu)
    plt.plot(df['time'], df['filtered_x'], 'b-', linewidth=2, label='Filtered Position (Kalman/Low-Pass)')
    
    # Personalizzazione assi e titoli per LaTeX
    plt.title('Filter Tuning: System Response along X-Axis', fontsize=14, fontweight='bold')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Pixel Coordinate [u]', fontsize=12)
    
    # Limiti asse Y (opzionale, per centrare meglio il grafico sulla risoluzione della camera)
    plt.ylim(0, 640) 
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # Salva il grafico ad alta risoluzione per la tesi
    plt.savefig('tuning_response_x.pdf', format='pdf', dpi=300)
    plt.show()
# AND AT THE END OF THE SCRIPT, CALL:
# plot_results(log_data)