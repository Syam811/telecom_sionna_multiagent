import os
import numpy as np
import matplotlib.pyplot as plt
from core.sionna_compat import phy_imports

def simulate_ber(
    modulation: str = "qpsk",
    channel: str = "awgn",
    snr_db_list=None,
    n_bits: int = 200000,
    batch_size: int = 2000,
    out_dir: str = "outputs"
):
    os.makedirs(out_dir, exist_ok=True)
    if snr_db_list is None:
        snr_db_list = [-5, 0, 5, 10, 15]

    try:
        import tensorflow as tf
        Constellation, Mapper, Demapper, AWGN, FlatFadingChannel, ebnodb2no = phy_imports()
    except Exception as e:
        return {"plots": [], "kpis": {}, "error": f"Sionna/TensorFlow import failed: {e}"}

    mod = modulation.lower()
    if mod == "qpsk":
        constellation = Constellation("qam", 4)
    elif "qam" in mod:
        m = int(mod.replace("qam", ""))
        constellation = Constellation("qam", m)
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown modulation: {modulation}"}

    mapper = Mapper(constellation)
    demapper = Demapper("app", constellation)

    fading = (channel.lower() == "rayleigh")
    if fading:
        channel_layer = FlatFadingChannel(num_tx_ant=1, num_rx_ant=1, add_awgn=True)
    else:
        channel_layer = AWGN()

    n_bits_per_sym = int(np.log2(constellation.num_points))
    bers = []

    for snr_db in snr_db_list:
        no = ebnodb2no(snr_db, n_bits_per_sym, coderate=1.0)
        n_err, n_tot = 0, 0

        while n_tot < n_bits:
            b = tf.random.uniform([batch_size, n_bits_per_sym], 0, 2, dtype=tf.int32)
            x = mapper(b)

            if fading:
                y, h = channel_layer([x, no])
                llr = demapper([y, h, no])
            else:
                y = channel_layer([x, no])
                llr = demapper([y, no])

            b_hat = tf.cast(llr > 0, tf.int32)
            n_err += tf.reduce_sum(tf.cast(tf.not_equal(b, b_hat), tf.int32)).numpy()
            n_tot += batch_size * n_bits_per_sym

        bers.append(n_err / n_tot)

    fig = plt.figure()
    plt.semilogy(snr_db_list, bers, marker="o")
    plt.title(f"BER vs SNR ({modulation.upper()} - {channel.upper()})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both")

    plot_path = os.path.join(out_dir, f"ber_{mod}_{channel}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plots": [plot_path],
        "kpis": {"snr_db": snr_db_list, "ber": bers, "modulation": modulation, "channel": channel}
    }
