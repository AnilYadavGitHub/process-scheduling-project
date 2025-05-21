import heapq
import matplotlib.pyplot as plt
from collections import deque

class Process:
    def __init__(self, pid, arrival, burst, priority=0):
        self.pid = pid
        self.arrival = arrival
        self.burst = burst
        self.remaining = burst
        self.completion = 0
        self.turnaround = 0
        self.waiting = 0
        self.response = -1
        self.energy_used = 0
        self.priority = priority  # For priority scheduling
        self.queue_level = 0      # For MLFQ

class EnergyAwareScheduler:
    def __init__(self):
        # Energy parameters (configurable)
        self.idle_power = 0.1    # Power consumption in idle state
        self.active_power = 1.0  # Power consumption when executing
        self.switch_power = 0.2  # Energy cost for context switching
        self.dvfs_levels = [     # Dynamic Voltage/Frequency Scaling levels
            {'freq': 1.0, 'power': 1.0, 'speed': 1.0},  # Full power
            {'freq': 0.8, 'power': 0.6, 'speed': 0.8},   # Medium power
            {'freq': 0.6, 'power': 0.4, 'speed': 0.6},   # Low power
        ]
    
    def get_user_input(self):
        """Get process information from user"""
        processes = []
        n = int(input("Enter number of processes: "))
        for i in range(n):
            print(f"\nProcess {i+1}:")
            arrival = int(input("  Arrival time: "))
            burst = int(input("  Burst time: "))
            priority = int(input(" Priority of process: "))
            processes.append(Process(i+1, arrival, burst, priority))
        quantum = int(input("\nEnter time quantum for Round Robin: "))
        return processes, quantum
    
    def fcfs(self, processes):
        """First Come First Serve with energy tracking"""
        if not processes:
            return [], [], 0
            
        processes.sort(key=lambda x: x.arrival)
        time = 0
        gantt = []
        total_energy = 0
        last_switch = 0
        
        for p in processes:
            # Handle idle time
            if time < p.arrival:
                idle_duration = p.arrival - time
                gantt.append(("Idle", time, p.arrival))
                total_energy += idle_duration * self.idle_power
                time = p.arrival
            
            # Context switch energy cost
            if time > last_switch:
                total_energy += self.switch_power
                last_switch = time
            
            # Process execution
            p.response = time - p.arrival
            exec_time = p.burst
            gantt.append((p.pid, time, time + exec_time))
            
            # Energy calculation
            energy = exec_time * self.active_power
            p.energy_used = energy
            total_energy += energy
            
            time += exec_time
            p.completion = time
            p.turnaround = p.completion - p.arrival
            p.waiting = p.turnaround - p.burst
        
        return processes, gantt, total_energy

    def sjf_energy_aware(self, processes):
        """Shortest Job First with energy optimization"""
        if not processes:
            return [], [], 0
            
        processes.sort(key=lambda x: x.arrival)
        time = 0
        completed = []
        gantt = []
        total_energy = 0
        last_switch = 0
        remaining_processes = processes.copy()
        
        while remaining_processes:
            ready_queue = [p for p in remaining_processes if p.arrival <= time]
            
            if not ready_queue:
                next_arrival = min(p.arrival for p in remaining_processes)
                idle_duration = next_arrival - time
                gantt.append(("Idle", time, next_arrival))
                total_energy += idle_duration * self.idle_power
                time = next_arrival
                continue
            
            ready_queue.sort(key=lambda x: x.burst)
            p = ready_queue[0]
            remaining_processes.remove(p)
            
            # Context switch energy cost
            if time > last_switch:
                total_energy += self.switch_power
                last_switch = time
            
            # Process execution with adaptive frequency
            p.response = time - p.arrival
            exec_time = p.burst
            
            # Energy-efficient decision: use lower frequency for non-critical processes
            if len(ready_queue) > 2:  # Many waiting processes
                level = self.dvfs_levels[0]  # Full speed
            else:
                level = self.dvfs_levels[2]  # Reduced speed
            
            actual_time = exec_time / level['speed']
            gantt.append((p.pid, time, time + actual_time))
            
            energy = actual_time * level['power']
            p.energy_used = energy
            total_energy += energy
            
            time += actual_time
            p.completion = time
            p.turnaround = p.completion - p.arrival
            p.waiting = p.turnaround - p.burst
            completed.append(p)
        
        return completed, gantt, total_energy

    def sjrf_energy_aware(self, processes):
        """Shortest Job Remaining First with energy optimization"""
        if not processes:
            return [], [], 0
            
        processes = sorted(processes, key=lambda x: x.arrival)
        time = 0
        completed = []
        gantt = []
        total_energy = 0
        last_switch = 0
        remaining_processes = [Process(p.pid, p.arrival, p.burst) for p in processes]
        ready_queue = []
        i = 0  # Track next process to arrive
        
        while i < len(remaining_processes) or ready_queue:
            # Add arriving processes
            while i < len(remaining_processes) and remaining_processes[i].arrival <= time:
                heapq.heappush(ready_queue, (remaining_processes[i].remaining, remaining_processes[i]))
                i += 1
            
            if not ready_queue:
                if i < len(remaining_processes):
                    next_arrival = remaining_processes[i].arrival
                    idle_duration = next_arrival - time
                    gantt.append(("Idle", time, next_arrival))
                    total_energy += idle_duration * self.idle_power
                    time = next_arrival
                continue
            
            # Get process with shortest remaining time
            _, current = heapq.heappop(ready_queue)
            
            # Context switch energy cost
            if time > last_switch:
                total_energy += self.switch_power
                last_switch = time
            
            if current.response == -1:
                current.response = time - current.arrival
            
            # Determine next event (arrival or completion)
            next_event = remaining_processes[i].arrival if i < len(remaining_processes) else float('inf')
            
            # Execute until next event or completion
            exec_time = min(current.remaining, next_event - time) if next_event != float('inf') else current.remaining
            
            # Adaptive frequency scaling
            if len(ready_queue) > 2:
                level = self.dvfs_levels[0]  # Full speed for busy system
            else:
                level = self.dvfs_levels[1]  # Medium speed
            
            actual_time = exec_time / level['speed']
            gantt.append((current.pid, time, time + actual_time))
            
            energy = actual_time * level['power']
            current.energy_used += energy
            total_energy += energy
            
            current.remaining -= exec_time
            time += actual_time
            
            if current.remaining > 0:
                heapq.heappush(ready_queue, (current.remaining, current))
            else:
                current.completion = time
                current.turnaround = current.completion - current.arrival
                current.waiting = current.turnaround - current.burst
                completed.append(current)
        
        return completed, gantt, total_energy

    def priority_queue_scheduling(self, processes):
        """Priority Queue Scheduling with energy awareness"""
        if not processes:
            return [], [], 0
            
        processes = sorted(processes, key=lambda x: x.arrival)
        time = 0
        completed = []
        gantt = []
        total_energy = 0
        last_switch = 0
        remaining_processes = [Process(p.pid, p.arrival, p.burst, p.priority) for p in processes]
        ready_queue = []
        i = 0  # Track next process to arrive
        
        while i < len(remaining_processes) or ready_queue:
            # Add arriving processes
            while i < len(remaining_processes) and remaining_processes[i].arrival <= time:
                heapq.heappush(ready_queue, (remaining_processes[i].priority, remaining_processes[i]))
                i += 1
            
            if not ready_queue:
                if i < len(remaining_processes):
                    next_arrival = remaining_processes[i].arrival
                    idle_duration = next_arrival - time
                    gantt.append(("Idle", time, next_arrival))
                    total_energy += idle_duration * self.idle_power
                    time = next_arrival
                continue
            
            # Get highest priority process
            _, current = heapq.heappop(ready_queue)
            
            # Context switch energy cost
            if time > last_switch:
                total_energy += self.switch_power
                last_switch = time
            
            if current.response == -1:
                current.response = time - current.arrival
            
            # Execute entire burst (non-preemptive)
            exec_time = current.burst
            
            # Energy-efficient frequency selection based on priority
            if current.priority <= 2:  # High priority
                level = self.dvfs_levels[0]  # Full speed
            else:
                level = self.dvfs_levels[2]  # Low speed
            
            actual_time = exec_time / level['speed']
            gantt.append((current.pid, time, time + actual_time))
            
            energy = actual_time * level['power']
            current.energy_used = energy
            total_energy += energy
            
            time += actual_time
            current.completion = time
            current.turnaround = current.completion - current.arrival
            current.waiting = current.turnaround - current.burst
            completed.append(current)
        
        return completed, gantt, total_energy

    def mlfq_energy_aware(self, processes, quantum):
        """Multilevel Feedback Queue with energy optimization"""
        if not processes:
            return [], [], 0
            
        # Create queues (3 levels)
        queues = [
            deque(),  # Highest priority, RR with full quantum
            deque(),  # Medium priority, RR with reduced quantum
            deque()   # Lowest priority, FCFS
        ]
        
        processes = sorted(processes, key=lambda x: x.arrival)
        time = 0
        completed = []
        gantt = []
        total_energy = 0
        last_switch = 0
        remaining_processes = [Process(p.pid, p.arrival, p.burst) for p in processes]
        i = 0  # Track next process to arrive
        
        while i < len(remaining_processes) or any(queues):
            # Add arriving processes to top queue
            while i < len(remaining_processes) and remaining_processes[i].arrival <= time:
                queues[0].append(remaining_processes[i])
                i += 1
            
            # Find highest priority non-empty queue
            current_queue = next((q for q in queues if q), None)
            
            if not current_queue:
                if i < len(remaining_processes):
                    next_arrival = remaining_processes[i].arrival
                    idle_duration = next_arrival - time
                    gantt.append(("Idle", time, next_arrival))
                    total_energy += idle_duration * self.idle_power
                    time = next_arrival
                continue
            
            # Get process from current queue
            current = current_queue.popleft()
            queue_level = queues.index(current_queue)
            
            # Context switch energy cost
            if time > last_switch:
                total_energy += self.switch_power
                last_switch = time
            
            if current.response == -1:
                current.response = time - current.arrival
            
            # Determine time slice based on queue level
            if queue_level == 0:
                time_slice = min(quantum, current.remaining)
            elif queue_level == 1:
                time_slice = min(quantum // 2, current.remaining)
            else:  # FCFS
                time_slice = current.remaining
            
            # Select frequency level based on queue
            level = self.dvfs_levels[2 - queue_level]  # Higher queue = higher frequency
            
            actual_time = time_slice / level['speed']
            gantt.append((current.pid, time, time + actual_time))
            
            energy = actual_time * level['power']
            current.energy_used += energy
            total_energy += energy
            
            current.remaining -= time_slice
            time += actual_time
            
            # Check if process completed
            if current.remaining == 0:
                current.completion = time
                current.turnaround = current.completion - current.arrival
                current.waiting = current.turnaround - current.burst
                completed.append(current)
            else:
                # Demote process to lower queue if not completed
                new_level = min(queue_level + 1, 2)
                queues[new_level].append(current)
        
        return completed, gantt, total_energy

    def round_robin_eco(self, processes, quantum):
        """Energy-aware Round Robin with dynamic quantum adjustment"""
        if not processes:
            return [], [], 0
            
        processes = sorted(processes, key=lambda x: x.arrival)
        time = 0
        queue = deque()
        completed = []
        gantt = []
        total_energy = 0
        last_switch = 0
        remaining_processes = processes.copy()
        i = 0  # Track next process to arrive
        
        while queue or i < len(remaining_processes):
            # Add arriving processes
            while i < len(remaining_processes) and remaining_processes[i].arrival <= time:
                queue.append(remaining_processes[i])
                i += 1
            
            if not queue:
                if i < len(remaining_processes):
                    next_arrival = remaining_processes[i].arrival
                    idle_duration = next_arrival - time
                    gantt.append(("Idle", time, next_arrival))
                    total_energy += idle_duration * self.idle_power
                    time = next_arrival
                continue
            
            p = queue.popleft()
            
            # Context switch energy cost
            if time > last_switch:
                total_energy += self.switch_power
                last_switch = time
            
            if p.response == -1:
                p.response = time - p.arrival
            
            # Dynamic quantum adjustment
            current_quantum = min(quantum, p.remaining)
            
            # Select frequency level based on queue length
            if len(queue) > 3:  # Many waiting processes
                level = self.dvfs_levels[0]  # Full speed
            else:
                level = self.dvfs_levels[1]  # Medium speed
            
            actual_time = current_quantum / level['speed']
            gantt.append((p.pid, time, time + actual_time))
            
            energy = actual_time * level['power']
            p.energy_used += energy
            total_energy += energy
            
            p.remaining -= current_quantum
            time += actual_time
            
            # Add processes that arrived during this quantum
            while i < len(remaining_processes) and remaining_processes[i].arrival <= time:
                queue.append(remaining_processes[i])
                i += 1
            
            if p.remaining > 0:
                queue.append(p)
            else:
                p.completion = time
                p.turnaround = p.completion - p.arrival
                p.waiting = p.turnaround - p.burst
                completed.append(p)
        
        return completed, gantt, total_energy

    def calculate_metrics(self, processes, total_energy):
        """Calculate performance metrics"""
        if not processes:
            return {}
            
        total_time = max(p.completion for p in processes) if processes else 0
        return {
            'avg_turnaround': sum(p.turnaround for p in processes) / len(processes),
            'avg_waiting': sum(p.waiting for p in processes) / len(processes),
            'avg_response': sum(p.response for p in processes) / len(processes),
            'throughput': len(processes) / total_time if total_time > 0 else 0,
            'total_energy': total_energy,
            'power_efficiency': len(processes) / total_energy if total_energy > 0 else 0
        }

    def display_process_table(self, processes, title):
        """Display detailed process table with metrics"""
        print(f"\n{title} Process Table:")
        print("+-----+----+----+----+-----+----+----+")
        print("| PID | AT | BT | CT | TAT | WT | RT |")
        print("+-----+----+----+----+-----+----+----+")
        
        for p in sorted(processes, key=lambda x: x.pid):
            print(f"| {p.pid:3} | {p.arrival:2} | {p.burst:2} | {p.completion:2} | {p.turnaround:3} | {p.waiting:2} | {p.response:2} |")
        
        print("+-----+----+----+----+-----+----+----+")

    def plot_gantt_with_energy(self, title, gantt, total_energy):
        """Enhanced Gantt chart showing energy usage"""
        if not gantt:
            print(f"No data to plot for {title}")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
        
        # Process timeline
        for entry in gantt:
            pid, start, end = entry
            color = 'lightgreen' if pid != "Idle" else 'lightcoral'
            label = f"P{pid}" if pid != "Idle" else "Idle"
            ax1.barh(y=label, left=start, width=end-start, color=color, edgecolor='black')
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Process")
        ax1.set_title(f"{title} (Total Energy: {total_energy:.2f} units)")
        ax1.grid(axis='x', linestyle='--')
        
        # Energy timeline
        for entry in gantt:
            pid, start, end = entry
            duration = end - start
            power = self.idle_power if pid == "Idle" else self.active_power
            ax2.barh(y="Energy", left=start, width=duration, 
                    height=0.5, color='skyblue' if pid != "Idle" else 'lightpink', 
                    edgecolor='navy')
        
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Power State")
        ax2.grid(axis='x', linestyle='--')
        
        plt.tight_layout()
        plt.show()

    def run_comparison(self):
        """Main function to run the comparison"""
        print("=== Energy-Efficient Process Scheduling Simulator ===")
        print("Please enter process details:\n")
        
        processes, quantum = self.get_user_input()
        
        print("\nRunning schedulers...")
        
        # Run FCFS
        fcfs_procs = [Process(p.pid, p.arrival, p.burst) for p in processes]
        fcfs_results, fcfs_gantt, fcfs_energy = self.fcfs(fcfs_procs)
        fcfs_metrics = self.calculate_metrics(fcfs_results, fcfs_energy)
        
        # Run SJF Energy Aware
        sjf_procs = [Process(p.pid, p.arrival, p.burst) for p in processes]
        sjf_results, sjf_gantt, sjf_energy = self.sjf_energy_aware(sjf_procs)
        sjf_metrics = self.calculate_metrics(sjf_results, sjf_energy)
        
        # Run SJRF Energy Aware
        sjrf_procs = [Process(p.pid, p.arrival, p.burst) for p in processes]
        sjrf_results, sjrf_gantt, sjrf_energy = self.sjrf_energy_aware(sjrf_procs)
        sjrf_metrics = self.calculate_metrics(sjrf_results, sjrf_energy)
        
        # Run Priority Queue
        pq_procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in processes]
        pq_results, pq_gantt, pq_energy = self.priority_queue_scheduling(pq_procs)
        pq_metrics = self.calculate_metrics(pq_results, pq_energy)
        
        # Run MLFQ
        mlfq_procs = [Process(p.pid, p.arrival, p.burst) for p in processes]
        mlfq_results, mlfq_gantt, mlfq_energy = self.mlfq_energy_aware(mlfq_procs, quantum)
        mlfq_metrics = self.calculate_metrics(mlfq_results, mlfq_energy)
        
        # Run Round Robin Eco
        rr_procs = [Process(p.pid, p.arrival, p.burst) for p in processes]
        rr_results, rr_gantt, rr_energy = self.round_robin_eco(rr_procs, quantum)
        rr_metrics = self.calculate_metrics(rr_results, rr_energy)
        
        # Display detailed process tables
        self.display_process_table(fcfs_results, "FCFS")
        self.display_process_table(sjf_results, "SJF Energy Aware")
        self.display_process_table(sjrf_results, "SJRF Energy Aware")
        self.display_process_table(pq_results, "Priority Queue")
        self.display_process_table(mlfq_results, f"MLFQ (Q={quantum})")
        self.display_process_table(rr_results, f"Round Robin (Q={quantum})")
        
        # Display results
        print("\n=== Scheduling Algorithm Comparison ===")
        print("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Algorithm", "Avg Turnaround", "Avg Waiting", "Throughput", "Total Energy", "Power Eff."))
        
        for name, metrics in [
            ("FCFS", fcfs_metrics),
            ("SJF Energy Aware", sjf_metrics),
            ("SJRF Energy Aware", sjrf_metrics),
            ("Priority Queue", pq_metrics),
            ("MLFQ", mlfq_metrics),
            ("RR Energy Aware", rr_metrics)
        ]:
            print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.4f}".format(
                name,
                metrics['avg_turnaround'],
                metrics['avg_waiting'],
                metrics['throughput'],
                metrics['total_energy'],
                metrics['power_efficiency']
            ))
        
        # Plot Gantt charts
        self.plot_gantt_with_energy("FCFS Scheduling", fcfs_gantt, fcfs_energy)
        self.plot_gantt_with_energy("SJF Energy Aware", sjf_gantt, sjf_energy)
        self.plot_gantt_with_energy("SJRF Energy Aware", sjrf_gantt, sjrf_energy)
        self.plot_gantt_with_energy("Priority Queue", pq_gantt, pq_energy)
        self.plot_gantt_with_energy(f"MLFQ (Q={quantum})", mlfq_gantt, mlfq_energy)
        self.plot_gantt_with_energy(f"Round Robin (Q={quantum})", rr_gantt, rr_energy)

if __name__ == "__main__":
    scheduler = EnergyAwareScheduler()
    scheduler.run_comparison()
