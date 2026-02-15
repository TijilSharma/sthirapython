from controller.hackpull import run_fetch

def initialize_ohlcv():
    print("Initializing OHLCV CSV...")
    run_fetch()  # calls the whole script as-is
    print("Initialization complete.")