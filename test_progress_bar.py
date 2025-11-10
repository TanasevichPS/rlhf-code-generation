"""
Quick test script to verify progress bars work correctly.
"""

import time
from tqdm import tqdm
import sys

print("\n" + "="*80)
print("TESTING PROGRESS BARS")
print("="*80)

# Test 1: Basic progress bar
print("\n[Test 1] Basic Progress Bar:")
for i in tqdm(range(100), desc="Basic", unit="item"):
    time.sleep(0.01)

# Test 2: Progress bar with postfix
print("\n[Test 2] Progress Bar with Live Metrics:")
pbar = tqdm(range(50), desc="Training", unit="batch")
for i in pbar:
    loss = 2.5 - i * 0.03
    reward = i * 0.02
    pbar.set_postfix({
        'loss': f'{loss:.4f}',
        'reward': f'{reward:.4f}',
        'step': f'{i}/50'
    })
    time.sleep(0.05)
pbar.close()

# Test 3: Nested progress bars
print("\n[Test 3] Epoch Progress (simulated):")
for epoch in range(3):
    pbar = tqdm(
        range(20),
        desc=f"Epoch {epoch+1}/3",
        unit="batch",
        bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        leave=True,
        position=0
    )
    for batch in pbar:
        pbar.set_postfix({'loss': f'{2.0/(epoch+1):.4f}', 'reward': f'{epoch*0.3:.2f}'})
        time.sleep(0.05)
    pbar.close()
    print(f"  Epoch {epoch+1} completed!")

print("\n" + "="*80)
print("✅ All progress bar tests completed!")
print("="*80 + "\n")

