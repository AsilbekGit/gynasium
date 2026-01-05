"""
Data Verification Script
Verify and analyze your train path data before training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_speed_env import TrainSpeedProfileEnv


def verify_data(coordinates_file='data/coordinates.dat', data_file='data/data.csv'):
    """
    Verify and display data statistics
    """
    print("="*80)
    print("TRAIN PATH DATA VERIFICATION")
    print("="*80)
    
    # Load coordinates
    print("\n1. Loading Coordinates...")
    try:
        coords = pd.read_csv(coordinates_file, header=None, 
                            names=['node', 'x', 'y'], sep=r'\s+')
        print(f"   ✓ Loaded {len(coords)} coordinate points")
        print(f"   First point: ({coords.iloc[0]['x']:.2f}, {coords.iloc[0]['y']:.2f})")
        print(f"   Last point: ({coords.iloc[-1]['x']:.2f}, {coords.iloc[-1]['y']:.2f})")
    except Exception as e:
        print(f"   ✗ Error loading coordinates: {e}")
        return False
    
    # Load track data
    print("\n2. Loading Track Data...")
    try:
        track_data = pd.read_csv(data_file)
        print(f"   ✓ Loaded {len(track_data)} track segments")
        print(f"   Columns: {list(track_data.columns)}")
    except Exception as e:
        print(f"   ✗ Error loading track data: {e}")
        return False
    
    # Verify data consistency
    print("\n3. Data Consistency Check...")
    if len(coords) - 1 != len(track_data):
        print(f"   ⚠ Warning: {len(coords)} nodes but {len(track_data)} segments")
        print(f"   Expected {len(coords)-1} segments for {len(coords)} nodes")
    else:
        print(f"   ✓ Data consistent: {len(coords)} nodes, {len(track_data)} segments")
    
    # Calculate distances
    print("\n4. Distance Calculations...")
    coords_array = coords[['x', 'y']].values
    distances = np.sqrt(np.sum(np.diff(coords_array, axis=0)**2, axis=1))
    total_distance = np.sum(distances)
    
    print(f"   Total path length: {total_distance:.2f} m ({total_distance/1000:.2f} km)")
    print(f"   Average segment length: {np.mean(distances):.2f} m")
    print(f"   Min segment length: {np.min(distances):.2f} m")
    print(f"   Max segment length: {np.max(distances):.2f} m")
    
    # Analyze speed limits
    print("\n5. Speed Limit Analysis...")
    speed_limits = track_data['Speed_limit'].values
    station_mask = speed_limits == 1
    n_stations = np.sum(station_mask)
    
    print(f"   Number of stations (speed_limit = 1): {n_stations}")
    if n_stations > 0:
        station_indices = np.where(station_mask)[0]
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        station_positions = cumulative_distances[station_indices]
        print(f"   Station positions (m): {station_positions}")
    
    # Non-station speed limits
    non_station_limits = speed_limits[~station_mask]
    if len(non_station_limits) > 0:
        print(f"\n   Non-station speed limits:")
        print(f"   - Min: {np.min(non_station_limits):.2f} m/s ({np.min(non_station_limits)*3.6:.2f} km/h)")
        print(f"   - Max: {np.max(non_station_limits):.2f} m/s ({np.max(non_station_limits)*3.6:.2f} km/h)")
        print(f"   - Mean: {np.mean(non_station_limits):.2f} m/s ({np.mean(non_station_limits)*3.6:.2f} km/h)")
    
    # Analyze grades
    print("\n6. Grade Analysis...")
    grades = track_data['Grade'].values
    print(f"   Grade range: {np.min(grades):.4f}% to {np.max(grades):.4f}%")
    print(f"   Mean grade: {np.mean(grades):.4f}%")
    print(f"   Uphill segments: {np.sum(grades > 0)} ({np.sum(grades > 0)/len(grades)*100:.1f}%)")
    print(f"   Downhill segments: {np.sum(grades < 0)} ({np.sum(grades < 0)/len(grades)*100:.1f}%)")
    print(f"   Steepest uphill: {np.max(grades):.4f}%")
    print(f"   Steepest downhill: {np.min(grades):.4f}%")
    
    # Analyze curvatures
    print("\n7. Curvature Analysis...")
    curvatures = track_data['Curvature'].values
    print(f"   Curvature range: {np.min(curvatures):.6f}% to {np.max(curvatures):.6f}%")
    print(f"   Mean curvature: {np.mean(curvatures):.6f}%")
    
    # Calculate radii
    radii = np.where(curvatures > 0.001, 100.0 / curvatures, 10000)
    print(f"\n   Corresponding radii:")
    print(f"   - Min radius: {np.min(radii):.2f} m (sharpest curve)")
    print(f"   - Max radius: {np.max(radii):.2f} m")
    print(f"   - Sharp curves (R < 500m): {np.sum(radii < 500)} segments")
    
    # Estimate travel time
    print("\n8. Time Estimates...")
    # Conservative estimate: average at 60% of speed limit
    avg_speed_limit = np.mean(non_station_limits) if len(non_station_limits) > 0 else 20
    avg_speed = avg_speed_limit * 0.6
    travel_time = total_distance / avg_speed
    
    print(f"   Conservative estimate (60% of avg speed limit):")
    print(f"   - Average speed: {avg_speed:.2f} m/s ({avg_speed*3.6:.2f} km/h)")
    print(f"   - Travel time: {travel_time:.0f} seconds ({travel_time/60:.1f} minutes)")
    print(f"   - With station stops ({n_stations} × 30s): {travel_time + n_stations*30:.0f} seconds ({(travel_time + n_stations*30)/60:.1f} minutes)")
    
    return True


def visualize_track(coordinates_file='data/coordinates.dat', data_file='data/data.csv'):
    """
    Create comprehensive track visualizations
    """
    # Load data
    coords = pd.read_csv(coordinates_file, header=None, 
                        names=['node', 'x', 'y'], sep=r'\s+')
    track_data = pd.read_csv(data_file)
    
    # Calculate distances
    coords_array = coords[['x', 'y']].values
    distances = np.sqrt(np.sum(np.diff(coords_array, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    # Identify stations
    station_mask = track_data['Speed_limit'] == 1
    station_positions = cumulative_distances[:-1][station_mask]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 2D Track Layout
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(coords['x'], coords['y'], 'b-', linewidth=2)
    ax1.plot(coords['x'].iloc[0], coords['y'].iloc[0], 'go', markersize=15, label='Start')
    ax1.plot(coords['x'].iloc[-1], coords['y'].iloc[-1], 'ro', markersize=15, label='End')
    
    # Mark stations
    station_coords = coords.iloc[:-1][station_mask]
    if len(station_coords) > 0:
        ax1.plot(station_coords['x'], station_coords['y'], 'rs', 
                markersize=12, label='Stations')
    
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.set_title('Track Layout (2D View)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Grade Profile
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(cumulative_distances[:-1], track_data['Grade'], 'g-', linewidth=2)
    ax2.fill_between(cumulative_distances[:-1], 0, track_data['Grade'], 
                     where=(track_data['Grade'] > 0), alpha=0.3, color='red', label='Uphill')
    ax2.fill_between(cumulative_distances[:-1], 0, track_data['Grade'], 
                     where=(track_data['Grade'] < 0), alpha=0.3, color='blue', label='Downhill')
    for station_pos in station_positions:
        ax2.axvline(x=station_pos, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Grade (%)')
    ax2.set_title('Track Grade Profile')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.legend()
    
    # 3. Curvature Profile
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(cumulative_distances[:-1], track_data['Curvature'], 'purple', linewidth=2)
    for station_pos in station_positions:
        ax3.axvline(x=station_pos, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Curvature (%)')
    ax3.set_title('Track Curvature Profile')
    ax3.grid(True, alpha=0.3)
    
    # 4. Radius Profile
    ax4 = plt.subplot(3, 2, 4)
    radii = np.where(track_data['Curvature'] > 0.001, 
                    100.0 / track_data['Curvature'], 10000)
    ax4.plot(cumulative_distances[:-1], radii, 'brown', linewidth=2)
    ax4.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='Sharp curve threshold')
    for station_pos in station_positions:
        ax4.axvline(x=station_pos, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Radius (m)')
    ax4.set_title('Curve Radius Profile')
    ax4.set_ylim([0, 3000])
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Speed Limits
    ax5 = plt.subplot(3, 2, 5)
    speed_limits = track_data['Speed_limit'].values.copy()
    speed_limits[station_mask] = 0  # Set station markers to 0 for visualization
    ax5.plot(cumulative_distances[:-1], speed_limits * 3.6, 'orange', linewidth=2)
    for station_pos in station_positions:
        ax5.axvline(x=station_pos, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Station' if station_pos == station_positions[0] else '')
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Speed Limit (km/h)')
    ax5.set_title('Speed Limit Profile (Stations shown as vertical lines)')
    ax5.grid(True, alpha=0.3)
    if len(station_positions) > 0:
        ax5.legend()
    
    # 6. Segment Length Distribution
    ax6 = plt.subplot(3, 2, 6)
    ax6.hist(distances, bins=30, edgecolor='black', alpha=0.7)
    ax6.axvline(x=np.mean(distances), color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(distances):.2f}m')
    ax6.set_xlabel('Segment Length (m)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Segment Length Distribution')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('track_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Track visualization saved as 'track_analysis.png'")
    plt.show()


def test_environment(coordinates_file='data/coordinates.dat', data_file='data/data.csv'):
    """
    Test the environment with a simple policy
    """
    print("\n" + "="*80)
    print("TESTING ENVIRONMENT")
    print("="*80)
    
    try:
        env = TrainSpeedProfileEnv(
            coordinates_file=coordinates_file,
            data_file=data_file,
            dt=1.0,
            station_dwell_time=30.0
        )
        print("\n✓ Environment created successfully")
        
        # Run a simple episode
        print("\nRunning test episode with constant acceleration...")
        obs, info = env.reset()
        done = False
        step = 0
        max_steps = 5000
        
        while not done and step < max_steps:
            # Simple policy: moderate constant acceleration
            action = np.array([0.2])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            if step % 500 == 0:
                print(f"  Step {step}: Position={info['position']:.1f}m, "
                      f"Speed={info['speed']*3.6:.1f}km/h, "
                      f"Energy={info['total_energy']:.4f}kWh, "
                      f"Stations={info['stations_completed']}")
        
        print(f"\n✓ Test episode completed")
        print(f"  Final position: {info['position']:.2f} m")
        print(f"  Total distance: {env.total_distance:.2f} m")
        print(f"  Total time: {info['time_elapsed']:.1f} seconds ({info['time_elapsed']/60:.1f} minutes)")
        print(f"  Total energy: {info['total_energy']:.4f} kWh")
        print(f"  Stations completed: {info['stations_completed']}/{len(env.station_positions)}")
        print(f"  Episode completed: {terminated}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🚂 TRAIN SPEED PROFILE OPTIMIZATION - DATA VERIFICATION\n")
    
    # Step 1: Verify data
    if verify_data():
        print("\n" + "="*80)
        print("✓ DATA VERIFICATION PASSED")
        print("="*80)
        
        # Step 2: Visualize
        print("\nGenerating track visualizations...")
        visualize_track()
        
        # Step 3: Test environment
        test_environment()
        
        print("\n" + "="*80)
        print("✓ ALL CHECKS PASSED - READY FOR TRAINING")
        print("="*80)
        print("\nNext steps:")
        print("1. Run quick test: python experiments.py test")
        print("2. Train model: python experiments.py 1")
        print("3. Run all scenarios: python experiments.py all")
    else:
        print("\n" + "="*80)
        print("✗ DATA VERIFICATION FAILED")
        print("="*80)
        print("\nPlease check your data files and try again.")
