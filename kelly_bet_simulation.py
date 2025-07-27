import random
import matplotlib.pyplot as plt
import numpy as np

def simulate_process(n_steps, x0, p, b, frac):
    X = [x0]
    success_list = []
    for _ in range(n_steps - 1):
        success = bernoulli(p)
        success_list.append(success)

        # Compute the next value based on the pay off and success in the step
        x_next = X[-1]*(1 +frac*(success*b-1))

        X.append(x_next)
    
    return X, success_list

def Kelly_single_bet(p, b):
    f = p - (1-p)/(b-1)
    return f

def bernoulli(p):
    return 1 if random.random() < p else 0  # Uniform distribution thresholded and made into bernoulli

def expected_growth(p,b, frac):
    return p*np.log(1+frac*(b-1))+(1-p)*np.log(1-frac)

def simulation_single_bet(n_steps, n_samples, x0, p, b, frac):

    samples_frac = []
    samples_kelly = []
    dist_data = []
    end_value_frac = []
    end_value_kelly = []

    kelly_bet = Kelly_single_bet(p,b)

    for _ in range(n_samples):
        sample_frac, dist = simulate_process(n_steps, x0, p, b, frac)
        sample_kelly, dist = simulate_process(n_steps, x0, p, b, kelly_bet)
        
        samples_frac.append(sample_frac)
        samples_kelly.append(sample_kelly)
        
        end_value_frac.append(sample_frac[-1])
        end_value_kelly.append(sample_kelly[-1])

        dist_data.append(dist)

    # Convert to NumPy arrays (if not already)
    end_value_frac = np.array(end_value_frac)
    end_value_kelly = np.array(end_value_kelly)

    # Choose percentile cutoffs
    lower_percentile = 1
    upper_percentile = 99

    # Filter Fractional Betting
    low_f, high_f = np.percentile(end_value_frac, [lower_percentile, upper_percentile])
    filtered_frac = end_value_frac[(end_value_frac >= low_f) & (end_value_frac <= high_f)]

    # Filter Kelly Betting
    low_k, high_k = np.percentile(end_value_kelly, [lower_percentile, upper_percentile])
    filtered_kelly = end_value_kelly[(end_value_kelly >= low_k) & (end_value_kelly <= high_k)]

    plt.hist(filtered_frac, bins=10, alpha=1, color='blue', label='Fractional Bet')
    plt.hist(filtered_kelly, bins=10, alpha=1, color='red', label='Kelly Bet')

    plt.title("Histogram of Final Wealth")
    plt.xlabel("Final Wealth")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)


    flat_dist = np.concatenate(dist_data)

    plt.figure(figsize=(10, 6))
    plt.hist(flat_dist, alpha=0.7)


    plt.title("Distribution of Per-Step Returns or Ratios")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
  


    # Create subplots: 1 row, 2 columns
    fig, axs = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    # Plot sample paths for fractional betting
    axs[0].set_title("Fractional Betting Sample Paths")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Wealth")
    #axs[0].set_ylim([0, 25000])

    for sample in samples_frac:
        axs[0].plot(sample, alpha=0.3)

    

    # Plot sample paths for Kelly betting
    axs[1].set_title("Kelly Betting Sample Paths")
    axs[1].set_xlabel("Step")
    #axs[1].set_ylim([0, 25000])  # same Y scale for fair comparison

    for sample in samples_kelly:
        axs[1].plot(sample, alpha=0.3)

    

    plt.tight_layout()
    plt.show()
    # plt.text(x=0.95, y=0.95, s=f"End Value: {end_value:.2f}",
    #         transform=plt.gca().transAxes,
    #         ha='right', va='top', fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # plt.show()

# Parameters
n_steps = int(40) # Number of steps in the process
n_samples = 1000 # Number of different samples to simulate

x0 = 1000 # Starting wealth
p = 6/37 # probability of win
b = 6.5 # payoff
frac = 0.5 # Fraction of wealth is wagered

print('Kelly fraction', Kelly_single_bet(p,b))
print('Expected growth rate at Kelly fraction :', expected_growth(p,b, Kelly_single_bet(p,b)))
print('Expected growth rate at f = ', frac,' : ', expected_growth(p,b, frac))

simulation_single_bet(n_steps, n_samples, x0, p, b, frac)

