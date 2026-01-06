"""
Experiment Runner for Different Optimization Scenarios
This script helps run experiments with different objectives and constraints
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_speed_env import TrainSpeedProfileEnv
from train_rl_training import train_agent, evaluate_agent
import json


class ExperimentRunner:
    """
    Run and manage multiple experiments with different configurations
    """
    
    def __init__(self, coordinates_file, data_file):
        self.coordinates_file = coordinates_file
        self.data_file = data_file
        self.experiments = []
    
    def create_env_variant(self, experiment_name, **kwargs):
        """
        Create environment variant with specific parameters
        """
        default_params = {
            'coordinates_file': self.coordinates_file,
            'data_file': self.data_file,
            'dt': 1.0
        }
        default_params.update(kwargs)
        
        env = TrainSpeedProfileEnv(**default_params)
        return env, experiment_name
    
    def run_experiment(self, experiment_name, env, algorithm='PPO', 
                      timesteps=500000, n_eval_episodes=10):
        """
        Run a single experiment
        
        Args:
            experiment_name: Name of the experiment
            env: UNWRAPPED Gymnasium environment
            algorithm: RL algorithm to use
            timesteps: Training timesteps
            n_eval_episodes: Number of evaluation episodes
        """
        import os
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        # Create experiment directory
        exp_dir = f'./experiments/{experiment_name}/'
        os.makedirs(exp_dir, exist_ok=True)
        
        # Train (train_agent will wrap the env internally)
        model, metrics = train_agent(
            env,
            algorithm=algorithm,
            total_timesteps=timesteps,
            save_path=exp_dir,
            log_path=f'./logs/{experiment_name}/'
        )
        
        # Evaluate (use unwrapped env)
        eval_results = evaluate_agent(model, env, n_episodes=n_eval_episodes)
        
        # Store results (convert numpy types to Python types for JSON serialization)
        result = {
            'name': experiment_name,
            'algorithm': algorithm,
            'timesteps': timesteps,
            'mean_energy': float(eval_results['mean_energy']),
            'std_energy': float(eval_results['std_energy']),
            'mean_time': float(eval_results['mean_time']),
            'std_time': float(eval_results['std_time']),
            'mean_reward': float(eval_results['mean_reward']),
            'speeds': [float(x) for x in eval_results['speeds'][0]],  # Convert to Python floats
            'positions': [float(x) for x in eval_results['positions'][0]]
        }
        
        self.experiments.append(result)
        
        # Save results
        with open(f'{exp_dir}/results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def compare_experiments(self, save_path='experiment_comparison.png'):
        """
        Compare all experiments
        """
        if not self.experiments:
            print("No experiments to compare!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = [exp['name'] for exp in self.experiments]
        energies = [exp['mean_energy'] for exp in self.experiments]
        energy_stds = [exp['std_energy'] for exp in self.experiments]
        times = [exp['mean_time'] for exp in self.experiments]
        time_stds = [exp['std_time'] for exp in self.experiments]
        rewards = [exp['mean_reward'] for exp in self.experiments]
        
        # Energy comparison
        x = np.arange(len(names))
        axes[0, 0].bar(x, energies, yerr=energy_stds, capsize=5)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Energy (kWh)')
        axes[0, 0].set_title('Energy Consumption Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time comparison
        axes[0, 1].bar(x, times, yerr=time_stds, capsize=5, color='orange')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Travel Time Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward comparison
        axes[1, 0].bar(x, rewards, color='green')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Average Reward Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Speed profiles comparison
        for exp in self.experiments:
            if 'speeds' in exp and 'positions' in exp:
                speeds_kmh = np.array(exp['speeds']) * 3.6
                axes[1, 1].plot(exp['positions'], speeds_kmh, 
                              label=exp['name'], alpha=0.7, linewidth=2)
        
        axes[1, 1].set_xlabel('Position (m)')
        axes[1, 1].set_ylabel('Speed (km/h)')
        axes[1, 1].set_title('Speed Profile Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary table
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"{'Experiment':<30} {'Energy (kWh)':<15} {'Time (s)':<12} {'Reward':<12}")
        print("-"*80)
        for exp in self.experiments:
            print(f"{exp['name']:<30} "
                  f"{exp['mean_energy']:>6.4f} ± {exp['std_energy']:>5.4f}    "
                  f"{exp['mean_time']:>6.1f} ± {exp['std_time']:>5.1f}   "
                  f"{exp['mean_reward']:>8.2f}")
        print("="*80)


def scenario_1_minimize_energy():
    """
    Scenario 1: Minimize energy with no time constraints
    Focus purely on energy efficiency
    """
    runner = ExperimentRunner('data/coordinates.dat', 'data/data.csv')
    
    env, name = runner.create_env_variant(
        'minimize_energy',
        target_time=None  # No time constraint
    )
    
    # Modify reward to focus heavily on energy
    result = runner.run_experiment(
        name,
        env,
        algorithm='PPO',
        timesteps=500000
    )
    
    return runner


def scenario_2_time_constrained():
    """
    Scenario 2: Minimize energy with fixed arrival time
    Balance between energy and schedule
    """
    runner = ExperimentRunner('data/coordinates.dat', 'data/data.csv')
    
    # Calculate reasonable target time (e.g., average speed of 25 m/s)
    env_temp = TrainSpeedProfileEnv('data/coordinates.dat', 'data/data.csv')
    target_time = env_temp.total_distance / 25  # seconds
    
    env, name = runner.create_env_variant(
        'time_constrained',
        target_time=target_time,
        target_time_tolerance=0.02  # 2% tolerance
    )
    
    result = runner.run_experiment(
        name,
        env,
        algorithm='PPO',
        timesteps=500000
    )
    
    return runner


def scenario_3_comfort_optimized():
    """
    Scenario 3: Optimize for passenger comfort
    Minimize jerky movements while maintaining efficiency
    """
    runner = ExperimentRunner('data/coordinates.dat', 'data/data.csv')
    
    # Need to modify reward function to emphasize comfort
    # This would require subclassing the environment
    # For now, using smaller time steps for smoother control
    
    env, name = runner.create_env_variant(
        'comfort_optimized',
        dt=0.5,  # Smaller time step for smoother control
        target_time=None
    )
    
    result = runner.run_experiment(
        name,
        env,
        algorithm='SAC',  # SAC often better for smooth control
        timesteps=500000
    )
    
    return runner


def scenario_4_compare_algorithms():
    """
    Scenario 4: Compare different RL algorithms
    """
    runner = ExperimentRunner('data/coordinates.dat', 'data/data.csv')
    
    algorithms = ['PPO', 'DDPG', 'SAC']
    
    for algo in algorithms:
        env, _ = runner.create_env_variant(f'algorithm_{algo}')
        runner.run_experiment(
            f'algorithm_{algo}',
            env,
            algorithm=algo,
            timesteps=300000  # Shorter for comparison
        )
    
    return runner


def scenario_5_adaptive_timestep():
    """
    Scenario 5: Test different control frequencies
    """
    runner = ExperimentRunner('data/coordinates.dat', 'data/data.csv')
    
    timesteps = [0.5, 1.0, 2.0]  # seconds
    
    for dt in timesteps:
        env, _ = runner.create_env_variant(
            f'timestep_{dt}s',
            dt=dt
        )
        runner.run_experiment(
            f'timestep_{dt}s',
            env,
            algorithm='PPO',
            timesteps=400000
        )
    
    return runner


def run_all_scenarios():
    """
    Run all optimization scenarios and compare
    """
    scenarios = {
        '1. Minimize Energy': scenario_1_minimize_energy,
        '2. Time Constrained': scenario_2_time_constrained,
        '3. Comfort Optimized': scenario_3_comfort_optimized,
        '4. Algorithm Comparison': scenario_4_compare_algorithms,
        '5. Adaptive Timestep': scenario_5_adaptive_timestep
    }
    
    print("\n" + "="*80)
    print("TRAIN SPEED PROFILE OPTIMIZATION - ALL SCENARIOS")
    print("="*80)
    
    all_runners = []
    
    for name, scenario_func in scenarios.items():
        print(f"\nStarting: {name}")
        try:
            runner = scenario_func()
            all_runners.append((name, runner))
            runner.compare_experiments(save_path=f'results_{name.replace(" ", "_")}.png')
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    # Create combined comparison if multiple scenarios completed
    if len(all_runners) > 1:
        print("\nCreating combined comparison...")
        combined_runner = ExperimentRunner('data/coordinates.dat', 'data/data.csv')
        for _, runner in all_runners:
            combined_runner.experiments.extend(runner.experiments)
        combined_runner.compare_experiments(save_path='all_scenarios_comparison.png')


def quick_test():
    """
    Quick test with minimal training for debugging
    """
    print("Running quick test...")
    
    runner = ExperimentRunner('data/coordinates.dat', 'data/data.csv')
    env, name = runner.create_env_variant('quick_test')
    
    result = runner.run_experiment(
        name,
        env,
        algorithm='PPO',
        timesteps=50000,  # Much shorter
        n_eval_episodes=3
    )
    
    print(f"\nQuick test complete!")
    print(f"Energy: {result['mean_energy']:.4f} kWh")
    print(f"Time: {result['mean_time']:.1f} s")
    
    return runner


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        
        if scenario == 'all':
            run_all_scenarios()
        elif scenario == 'test':
            quick_test()
        elif scenario == '1':
            runner = scenario_1_minimize_energy()
            runner.compare_experiments()
        elif scenario == '2':
            runner = scenario_2_time_constrained()
            runner.compare_experiments()
        elif scenario == '3':
            runner = scenario_3_comfort_optimized()
            runner.compare_experiments()
        elif scenario == '4':
            runner = scenario_4_compare_algorithms()
            runner.compare_experiments()
        elif scenario == '5':
            runner = scenario_5_adaptive_timestep()
            runner.compare_experiments()
        else:
            print(f"Unknown scenario: {scenario}")
            print("Usage: python experiments.py [all|test|1|2|3|4|5]")
    else:
        # Default: run quick test
        quick_test()