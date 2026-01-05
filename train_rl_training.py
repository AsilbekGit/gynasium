"""
Training and Analysis Script for Train Speed Profile Optimization
This script provides utilities for training, evaluating, and comparing RL agents
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from typing import List, Dict
import json


class TrainingCallback(BaseCallback):
    """
    Custom callback for logging training progress
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_energies = []
        self.episode_times = []
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            self.episode_rewards.append(self.locals.get('rewards')[0])
            self.episode_energies.append(info.get('total_energy', 0))
            self.episode_times.append(info.get('time_elapsed', 0))
            
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_energy = np.mean(self.episode_energies[-10:])
                print(f"Episodes: {len(self.episode_rewards)}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Energy: {avg_energy:.4f} kWh")
        
        return True


def train_agent(
    env,
    algorithm='PPO',
    total_timesteps=500000,
    save_path='./models/',
    log_path='./logs/',
    hyperparameters=None
):
    """
    Train an RL agent with specified algorithm and hyperparameters.
    
    Args:
        env: Gymnasium environment
        algorithm: 'PPO', 'DDPG', or 'SAC'
        total_timesteps: Total training timesteps
        save_path: Path to save models
        log_path: Path for tensorboard logs
        hyperparameters: Dict of algorithm-specific hyperparameters
    
    Returns:
        Trained model
    """
    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Optionally normalize observations
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {}
    
    # Create model based on algorithm
    if algorithm == 'PPO':
        default_params = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
        }
        default_params.update(hyperparameters)
        model = PPO("MlpPolicy", env, verbose=1, 
                   tensorboard_log=log_path, **default_params)
    
    elif algorithm == 'DDPG':
        default_params = {
            'learning_rate': 1e-3,
            'buffer_size': 200000,
            'learning_starts': 100,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.99,
        }
        default_params.update(hyperparameters)
        model = DDPG("MlpPolicy", env, verbose=1,
                    tensorboard_log=log_path, **default_params)
    
    elif algorithm == 'SAC':
        default_params = {
            'learning_rate': 3e-4,
            'buffer_size': 300000,
            'learning_starts': 100,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
        }
        default_params.update(hyperparameters)
        model = SAC("MlpPolicy", env, verbose=1,
                   tensorboard_log=log_path, **default_params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Setup callbacks
    training_callback = TrainingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix=f'{algorithm}_train_model'
    )
    
    # Train
    print(f"\nTraining {algorithm} agent for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[training_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{save_path}/{algorithm}_final")
    
    # Save training metrics
    metrics = {
        'rewards': training_callback.episode_rewards,
        'energies': training_callback.episode_energies,
        'times': training_callback.episode_times
    }
    
    with open(f"{save_path}/{algorithm}_metrics.json", 'w') as f:
        json.dump(metrics, f)
    
    return model, metrics


def evaluate_agent(model, env, n_episodes=10):
    """
    Evaluate a trained agent over multiple episodes.
    
    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []
    energies = []
    times = []
    speeds = []
    positions = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_speeds = []
        episode_positions = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_speeds.append(info['speed'])
            episode_positions.append(info['position'])
        
        rewards.append(episode_reward)
        energies.append(info['total_energy'])
        times.append(info['time_elapsed'])
        speeds.append(episode_speeds)
        positions.append(episode_positions)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_energy': np.mean(energies),
        'std_energy': np.std(energies),
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'speeds': speeds,
        'positions': positions,
        'rewards': rewards,
        'energies': energies,
        'times': times
    }


def compare_baseline(env, n_episodes=10):
    """
    Create a simple baseline by maintaining constant speed.
    This helps evaluate if RL agent actually learned something useful.
    """
    baseline_energies = []
    baseline_times = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        total_energy = 0
        
        # Simple strategy: maintain speed near average speed limit
        target_speed = 25  # m/s (~90 km/h)
        
        done = False
        while not done:
            current_speed = obs[1]
            # Simple proportional control
            if current_speed < target_speed:
                action = np.array([0.3])  # Accelerate
            else:
                action = np.array([-0.1])  # Light braking
            
            obs, reward, done, info = env.step(action)
            total_energy = info['total_energy']
        
        baseline_energies.append(total_energy)
        baseline_times.append(info['time_elapsed'])
    
    return {
        'mean_energy': np.mean(baseline_energies),
        'std_energy': np.std(baseline_energies),
        'mean_time': np.mean(baseline_times),
        'std_time': np.std(baseline_times)
    }


def plot_speed_profile(positions, speeds, track_data, save_path=None):
    """
    Plot the speed profile along with track characteristics.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Speed profile
    axes[0].plot(positions, np.array(speeds) * 3.6, 'b-', linewidth=2, label='Train Speed')
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title('Train Speed Profile')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Grade profile
    segment_positions = np.linspace(0, positions[-1], len(track_data))
    axes[1].plot(segment_positions, track_data['Grade'], 'g-', linewidth=2)
    axes[1].set_ylabel('Grade (%)')
    axes[1].set_title('Track Grade')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Curvature profile
    axes[2].plot(segment_positions, track_data['Curvature'], 'r-', linewidth=2)
    axes[2].set_ylabel('Curvature')
    axes[2].set_xlabel('Position (m)')
    axes[2].set_title('Track Curvature')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_progress(metrics_file):
    """
    Plot training metrics over time.
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Rewards
    episodes = range(len(metrics['rewards']))
    axes[0].plot(episodes, metrics['rewards'], alpha=0.6)
    
    # Smooth with moving average
    window = 50
    if len(metrics['rewards']) > window:
        smoothed = pd.Series(metrics['rewards']).rolling(window).mean()
        axes[0].plot(episodes, smoothed, 'r-', linewidth=2, label=f'MA({window})')
    
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training Progress: Rewards')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Energy consumption
    axes[1].plot(episodes, metrics['energies'], alpha=0.6)
    
    if len(metrics['energies']) > window:
        smoothed = pd.Series(metrics['energies']).rolling(window).mean()
        axes[1].plot(episodes, smoothed, 'r-', linewidth=2, label=f'MA({window})')
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Energy Consumption (kWh)')
    axes[1].set_title('Training Progress: Energy Efficiency')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def hyperparameter_search(env, algorithm='PPO', n_trials=10):
    """
    Simple random search for hyperparameters.
    """
    best_energy = float('inf')
    best_params = None
    results = []
    
    for trial in range(n_trials):
        print(f"\n=== Trial {trial + 1}/{n_trials} ===")
        
        # Random hyperparameters
        if algorithm == 'PPO':
            params = {
                'learning_rate': 10 ** np.random.uniform(-4, -2),
                'n_steps': np.random.choice([1024, 2048, 4096]),
                'batch_size': np.random.choice([32, 64, 128]),
                'n_epochs': np.random.choice([5, 10, 20]),
                'gamma': np.random.uniform(0.95, 0.999),
                'clip_range': np.random.uniform(0.1, 0.3),
            }
        
        # Train with these parameters
        model, metrics = train_agent(
            env,
            algorithm=algorithm,
            total_timesteps=100000,  # Shorter for search
            hyperparameters=params
        )
        
        # Evaluate
        eval_results = evaluate_agent(model, env, n_episodes=5)
        
        results.append({
            'params': params,
            'energy': eval_results['mean_energy'],
            'time': eval_results['mean_time'],
            'reward': eval_results['mean_reward']
        })
        
        if eval_results['mean_energy'] < best_energy:
            best_energy = eval_results['mean_energy']
            best_params = params
            print(f"New best! Energy: {best_energy:.4f} kWh")
    
    return best_params, results


# Example usage
if __name__ == "__main__":
    from train_speed_env import TrainSpeedProfileEnv
    
    # Create environment
    env = TrainSpeedProfileEnv(
        coordinates_file='coordinates.dat',
        data_file='data.csv',
        dt=1.0
    )
    
    # Option 1: Train a single agent
    print("Training PPO agent...")
    model, metrics = train_agent(
        env,
        algorithm='PPO',
        total_timesteps=500000,
        save_path='./models/',
        log_path='./logs/'
    )
    
    # Evaluate
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent(model, env, n_episodes=10)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Energy: {eval_results['mean_energy']:.4f} ± {eval_results['std_energy']:.4f} kWh")
    print(f"Mean Time: {eval_results['mean_time']:.1f} ± {eval_results['std_time']:.1f} s")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    # Compare with baseline
    print("\nBaseline comparison...")
    baseline = compare_baseline(env, n_episodes=10)
    print(f"Baseline Energy: {baseline['mean_energy']:.4f} ± {baseline['std_energy']:.4f} kWh")
    print(f"Energy Savings: {((baseline['mean_energy'] - eval_results['mean_energy']) / baseline['mean_energy'] * 100):.1f}%")
    
    # Plot speed profile
    plot_speed_profile(
        eval_results['positions'][0],
        eval_results['speeds'][0],
        env.track_data,
        save_path='speed_profile.png'
    )
    
    # Plot training progress
    plot_training_progress('./models/PPO_metrics.json')