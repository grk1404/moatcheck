from investment_alerts import check_tqqq_signal, send_whatsapp, CONFIG

info = check_tqqq_signal()
if info is None:
    print("check_tqqq_signal() returned None — see earlier logs for the error.")
else:
    lines = [
        "TQQQ Signal — " + info["signal"],
        "Regime: " + info["regime"],
        "Target: {:.2f}% TQQQ / {:.2f}% SQQQ".format(info["target_tqqq_pct"], info["target_sqqq_pct"]),
        "Sleeve value: ${:.2f}".format(info["account_value"]),
        "",
        "TQQQ: " + info["tqqq_action"],
        "SQQQ: " + info["sqqq_action"],
    ]
    message = "\n".join(lines)
    print("--- Message preview ---")
    print(message)
    print("--- Sending ---")
    ok = send_whatsapp(CONFIG["phone"], CONFIG["api_key"], message)
    print("Sent:", ok)
