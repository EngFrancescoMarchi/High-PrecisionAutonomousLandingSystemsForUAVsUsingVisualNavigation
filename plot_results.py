def plot_results(data):
    print(">>> GENERAZIONE GRAFICI IN CORSO... <<<")
    
    # Crea una figura con 3 grafici impilati
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Altitudine
    ax1.plot(data['time'], data['alt'], 'b-', label='Altitude')
    ax1.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('Mission Profile')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Errore di Posizione (Stima Kalman)
    ax2.plot(data['time'], data['pos_x_est'], 'r-', label='Error X (Est)')
    ax2.plot(data['time'], data['pos_y_est'], 'g-', label='Error Y (Est)')
    ax2.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
    ax2.set_ylabel('Position Error [m]') # O pixel se non hai convertito
    ax2.set_title('Tracking Error (Kalman Estimate)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Comandi di Velocità
    ax3.plot(data['time'], data['vel_x_cmd'], 'r--', label='Cmd Vel X')
    ax3.plot(data['time'], data['vel_y_cmd'], 'g--', label='Cmd Vel Y')
    
    # Aggiungi indicatore di visibilità target (area grigia quando perso)
    # Scaliamo per renderlo visibile
    vis = np.array(data['target_visible'])
    # Disegna sfondo rosso dove target persa
    # (Logica un po' complessa per matplotlib veloce, facciamo semplice:)
    # ax3.fill_between(data['time'], -1, 1, where=(vis==0), color='red', alpha=0.1, label='Target Lost')

    ax3.set_ylabel('Velocity Cmd [m/s]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Control Outputs')
    ax3.grid(True)
    ax3.legend()
    
    # Salva e Mostra
    plt.savefig('mission_log.png')
    print(">>> Grafico salvato come 'mission_log.png' <<<")
    plt.show()

# E ALLA FINE DELLO SCRIPT, CHIAMA:
# plot_results(log_data)