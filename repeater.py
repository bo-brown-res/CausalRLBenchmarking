import subprocess
import sys
import concurrent.futures

def run_command(cmd): # <-- ADD THIS FUNCTION
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return (cmd, result.stdout, result.returncode)
    except subprocess.CalledProcessError as e:
        return (cmd, e.stderr, e.returncode)

def run_experiments():
    # Loop through seeds 0 to 10

        # "--method_name", "TARNet",
        # // "--dataset_name", "mimic4_hourly"
        # // "--dataset_name", "epicare_len12_acts4ep_10000"
        # // "--dataset_name", "epicare_len72_acts4_vars32_eps25000",
        # "--dataset_name", "epicare_l48_a4_20v_deadcuredonly",
        # "--targets", "return", //'reward', 'return', '1-step-return'
        # "--target_value", "final_sum", //'binary', 'plusminusone', 'cumulative', 'reals', 'finals'
        # "--savetag", "outs_causal/final_sum+return",
        # "--reward_scaler", "1",
        # "--state_masking_p", "0.1",
        # "--random_seed", "10000"

    s = 3
    # t = 10
    # m_name = "DQN"
    # dname = "epicare_l48_a4_20v_deadcuredonly"
    dname = "epicare_l48_a4_20v_deadcuredonly_LOWBOOST"
    targets = "-step-return"
    targetvalue = "final_sum"

    commands = []
    for m_name in ["DQN", "CQL", "CausalDQN", "SoftActorCritic"]: #"DQN", "CQL", "CausalDQN", "SoftActorCritic"
        for state_masking_p in [1.0, 0.5]:
            for kstep in [1, 2, 4, 8, 16]:#range(1,1+t):#range(1, 10+1):
                # mask_p = px / 10
                savetag = f"outs_causal_lowboost/{kstep}{targets}_finsun+ret"

                for seed in range(10000, 10000+s):
                    # print(f"--- Starting run with random_seed={seed} ---")
                    command = [
                        sys.executable, "main.py", 
                        "--random_seed", str(seed),
                        "--state_masking_p", str(state_masking_p),
                        "--savetag", savetag,
                        ###########
                        "--method_name", m_name,
                        "--dataset_name", dname,
                        "--targets", f"{kstep}{targets}", 
                        "--target_value", targetvalue,
                            ]
                    commands.append(command)
                    # try:
                    #     subprocess.run(command, check=True)
                    #     print(f"--- Finished run {seed} successfully ---\n")
                        
                    # except subprocess.CalledProcessError as e:
                    #     print(f"!!! Run {seed} failed with error code {e.returncode} !!!\n")
                    # except KeyboardInterrupt:
                    #     print("\nStopping loop...")
                    #     break\
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(run_command, commands)

        # 5. Process results
        for cmd, stdout, code in results:
            print(f"✅ Command: {' '.join(cmd)}")
            print(f"   Output:\n{stdout.strip()}")
            print(f"   Return Code: {code}")

if __name__ == "__main__":
    run_experiments()