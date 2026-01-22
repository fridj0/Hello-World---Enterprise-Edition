# Hello World (the long way)

A deliberately terrible implementation of "Hello, world!"

## What It Does

Prints "Hello, world!" to the terminal after doing an absurd amount of unnecessary computation.

## Features

###  Exponential Fibonacci
- Calculates Fibonacci numbers up to the 25th term (by default) recursively
- **Never memoizes** - recalculates everything from scratch every time
- Used in garbage generation for maximum pain

###  Proof-of-Work Mining
- Each character requires finding a valid "nonce" (like in bitcoin miners)
- Mining difficulty **increases per character**: 8 bits → 20 bits
- Last character requires ~1 million hash attempts on average
- Inline assembly chaos injected every 1000/10000 attempts

###  Inline Assembly
- **CPUID spam** - Flushes CPU state 1000 times
- **Memory fences** - Kills CPU pipelining (mfence, lfence, sfence)
- **Cache thrashing** - Intentionally evicts 256KB from cache
- **RDTSC spinning** - Burns CPU cycles by reading timestamp counter

###  RNG Lottery System
- Every bit set has a chance to corrupt the byte instead
- Runs 20,000 RNG iterations + 128,000 hash iterations + 41,664 comparisons
- Checks all C(64,3) = 41,664 combinations of 3 bytes in two 64-byte hashes
- When collision found: prints "oops" and corrupts progress
- Caps at 50-100 disasters per run (randomized)

###  Garbage Generation
- Generates 32KB of "garbage" data using:
  - Exponential fibonacci (O(2^n) complexity)
  - 5 rounds of hashing per byte
  - Base conversion through 4 different number bases (7→13→23→31)
  - Memory leaks: 128 bytes every 100 bytes written
- **Validates by recomputing everything** (doubles the work!)
- Called on every validator attempt

###  Multi-threaded Chaos
- **Scribbler thread**: Races to corrupt your data while building
- **Validator thread**: Continuously checks if data is valid, generates garbage on every attempt
- **Main thread**: Builds message one bit at a time with mutex locks
- All threads fighting over shared state with maximum contention

###  YandereDev Else-If Tree
- **21 consecutive else-if branches** testing divisibility by primes
- Each branch does pointless volatile operations
- Maximum branch prediction chaos
- Features classics like `r + r - r` and `r * 2`

###  Memory Management
- Allocates with `mmap()` and never frees (inspired by Minecraft Medrock Edition)
- Small leaks scattered throughout (128 bytes per 100 bytes of garbage)
- Large leaks in validator (32KB + 16KB per attempt)
- 50% chance to crash with SIGSEGV when "freeing"

### 🔀 Additional Inefficiencies
- Bit-by-bit message construction with mutex locks
- Triple XOR operations for simple bit setting
- Pointless entropy addition and immediate removal
- Reconstructs characters bit-by-bit instead of reading directly
- Flushes stdout after every single character
- fsync() on stdout at the end
- Base conversions through multiple number systems
- Checksum computation the "long way" (bit by bit)
- If you get an "oops" the RNG lottery is currently broken so you're better off restarting it

## Compilation

```bash
gcc -Wall -Wextra -pedantic -o goodbyecruelworld goodbyecruelworld.c -lpthread
```

## Running

```bash
./goodbyecruelworld
```

Or with timing:
```bash
time ./goodbyecruelworld
```

## Expected Behavior

- Runtime: **30 seconds to several minutes** depending on:
  - How many "oops" disasters occur
  - Mining luck (finding nonces quickly vs slowly)
  - How many validator retries are needed
- CPU usage: **100%** on multiple cores
- Memory usage: **100MB - 3GB+** depending on validator attempts
- May print "oops" 0-100 times (random)
- **50% chance to segfault** at the end
- **Guaranteed** to print "Hello, world!" eventually (validator forces completion after 100k attempts)

## Why?

free will ig

## Architecture Notes

- Requires x86/x64 CPU (uses `__builtin_ia32_pause`, CPUID, RDTSC, cache flush instructions)
- Will not compile on ARM (Mac M1/M2/M3) without modifications
- POSIX systems only (Linux, macOS, BSD) (windows version soon)

## License

Public domain. Use this code however you want.

## Warning

This code is intentionally terrible. Do not use in production. Do not use anywhere. 
I am not responsible for any damage this script does to your system. Use at your own risk. (however, i have ran it myself and it hasn't wreaked havoc.. yet)
