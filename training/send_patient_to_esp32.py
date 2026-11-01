#!/usr/bin/env python3
"""
Sends a <record>_for_esp32.txt file (from export_patient_for_esp32.py) to the
ESP32 over Serial, line by line. Arduino IDE's own Serial Monitor can only
send one line at a time by hand, which is impractical for a few hundred
rows - this does it automatically with a small delay between lines so the
board's Serial buffer never overflows.

    pip install pyserial
    python send_patient_to_esp32.py COM5 b05_for_esp32.txt

Find your port name in Arduino IDE's Tools > Port menu, or Windows Device
Manager under "Ports (COM & LPT)".
"""
import sys
import time

import serial


def main():
    if len(sys.argv) < 3:
        print("usage: python send_patient_to_esp32.py <COM_PORT> <file.txt>")
        print("example: python send_patient_to_esp32.py COM5 b05_for_esp32.txt")
        sys.exit(1)
    port, path = sys.argv[1], sys.argv[2]

    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    print(f"loaded {len(lines)} lines from {path}")

    ser = serial.Serial(port, 115200, timeout=2)
    print(f"connected to {port}")

    # Opening the port auto-resets most ESP32 boards via DTR/RTS - rather than
    # hoping that happened cleanly (or that a manual EN-press lines up with
    # it), force a clean, known reset right here and discard anything already
    # sitting in the buffer first. This also removes any need to time a
    # manual EN-button press against running this script - just run it.
    ser.reset_input_buffer()
    ser.setDTR(False)
    time.sleep(0.2)
    ser.setDTR(True)
    print("forced a clean reset - waiting for the board to boot and print READY...")

    t0 = time.time()
    saw_ready = False
    while time.time() - t0 < 20:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print(f"  board: {line}")
        if "READY" in line:
            saw_ready = True
            break
    if not saw_ready:
        print("did not see READY within 20s after a forced reset - the board, port, or "
              "sketch may not be what's expected. Stopping rather than sending blindly.")
        ser.close()
        sys.exit(1)

    for i, line in enumerate(lines):
        ser.write((line + "\n").encode())
        time.sleep(0.01)   # small pacing delay - the board parses each line before the next arrives
        if i % 100 == 0:
            print(f"  sent {i}/{len(lines)} lines...")
        # drain any board output so its serial buffer doesn't back up
        while ser.in_waiting:
            print(f"  board: {ser.readline().decode(errors='replace').strip()}")

    print(f"sent all {len(lines)} lines.")
    print("=" * 60)
    print("LEAVE THIS WINDOW OPEN for the whole run. It now acts as the Serial")
    print("Monitor, printing everything the board says. Closing this script (or")
    print("opening the Arduino Serial Monitor) toggles the reset line and RESETS")
    print("THE BOARD, which wipes the patient data from its RAM - the board then")
    print("halts and silently drops out of the federation. Press Ctrl+C only once")
    print("the run is finished.")
    print("=" * 60)
    try:
        while True:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                print(f"  board: {line}", flush=True)
    except KeyboardInterrupt:
        print("\nstopping - the board will reset now that the port is closing.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
