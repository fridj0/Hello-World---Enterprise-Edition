#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <signal.h>
#include <time.h>
#include <stddef.h>
#include <math.h>
#include <complex.h>

#define EVER ;; // ohh yes
#define MAGIC 13
#define LOCK pthread_mutex_lock(&mb->mute);
#define UNLOCK pthread_mutex_unlock(&mb->mute);
#define MAYBE (rand()%100<50)
#define DEFINITELY 1
#define FIBONACCI_TERM 25

typedef struct MessageBuilder {
    volatile unsigned char *volatile codes;
    volatile int length;
    volatile _Bool locked;
    volatile _Bool scribbler_active;
    pthread_mutex_t mute;
    volatile unsigned long long checksum;
} MessageBuilder;

/* global state for extra coupling */
static MessageBuilder *volatile mb;
static volatile int print_ready = 0;
static volatile int disaster_count = 0;
static volatile int max_disasters = 25;

/* yes i have been learning assembly, can't you tell? */

#pragma once // alpha male vs sigma male vs #pragma male
#pragma GCC optimize("O0") // this is the only time the word "optimize" is used in the entire file, and it's to make it run slower

/* CPU feature detection - now cross-platform! */
static void waste_cpuid(void){
    for(int i=0;i<1000;i++){
#if defined(__x86_64__) || defined(__i386__)
        unsigned int eax, ebx, ecx, edx;
        __asm__ __volatile__(
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(0)
            : "memory"
        );
#elif defined(__aarch64__)
        /* ARM64: Read system registers - use EL0 accessible register */
        unsigned long ctr;
        __asm__ __volatile__("mrs %0, ctr_el0" : "=r"(ctr) :: "memory");
        (void)ctr;
#elif defined(__arm__)
        /* ARM32: Use MIDR from CP15 if available, otherwise just waste cycles */
#ifdef __ARM_ARCH_7A__
        unsigned long midr;
        __asm__ __volatile__("mrc p15, 0, %0, c0, c0, 0" : "=r"(midr) :: "memory");
        (void)midr;
#else
        volatile int x = i * i;
        (void)x;
#endif
#elif defined(__powerpc__) || defined(__ppc__)
        /* PowerPC: Read processor version register - SPR 287 */
        unsigned long pvr;
        __asm__ __volatile__("mfspr %0, 287" : "=r"(pvr) :: "memory");
        (void)pvr;
#else
        /* Generic waste */
        volatile int x = i * i;
        (void)x;
#endif
    }
}

/* Serialize execution, ruins CPU pipelining */
static void serialize_everything(void){
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__(
        "mfence\n\t"
        "lfence\n\t"
        "sfence\n\t"
        : : : "memory"
    );
#elif defined(__aarch64__) || defined(__arm__)
    /* ARM: Data Memory Barrier + Data Synchronization Barrier + Instruction Synchronization Barrier */
    __asm__ __volatile__("dmb sy" ::: "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
#elif defined(__powerpc__) || defined(__ppc__)
    /* PowerPC: sync + isync */
    __asm__ __volatile__("sync" ::: "memory");
    __asm__ __volatile__("isync" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/* put a brick wall every few feet on the highway */
/* evict data from the L1, L2 and L3 caches and make it walk all the way to system memory */
/* i had to read a lot of papers to get this in ppc32 */
static void trash_cache(void){
    volatile char buffer[256*1024]; // 256KB
    for(int i=0;i<256*1024;i+=64){
        buffer[i] = i;
#if defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("clflush (%0)" :: "r"(&buffer[i]) : "memory");
#elif defined(__aarch64__)
        /* ARM64: DC CIVAC (clean and invalidate by VA to PoC) */
        __asm__ __volatile__("dc civac, %0" :: "r"(&buffer[i]) : "memory");
#elif defined(__arm__)
        /* ARM32: Use available cache operations or just barrier */
#ifdef __ARM_ARCH_7A__
        __asm__ __volatile__("mcr p15, 0, %0, c7, c14, 1" :: "r"(&buffer[i]) : "memory");
#else
        __asm__ __volatile__("" ::: "memory");
#endif
#elif defined(__powerpc__) || defined(__ppc__)
        /* PowerPC: dcbf (data cache block flush) + sync to ensure completion */
        __asm__ __volatile__("dcbf 0, %0" :: "r"(&buffer[i]) : "memory");
        __asm__ __volatile__("sync" ::: "memory");
#else
        __sync_synchronize();
#endif
    }
}

/* Spin on timestamp counter, burns CPU cycles */
static void burn_cycles(unsigned long long cycles){
#if defined(__x86_64__) || defined(__i386__)
    unsigned int hi, lo, start_hi, start_lo;
    __asm__ __volatile__("rdtsc" : "=a"(start_lo), "=d"(start_hi) :: "memory");
    unsigned long long start = ((unsigned long long)start_hi << 32) | start_lo;
    unsigned long long now;
    do {
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi) :: "memory");
        now = ((unsigned long long)hi << 32) | lo;
    } while(now - start < cycles);
#elif defined(__aarch64__)
    /* ARM64: Use generic timer counter */
    unsigned long long start, now;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(start) :: "memory");
    do {
        __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(now) :: "memory");
    } while(now - start < cycles);
#elif defined(__arm__)
    /* ARM32: Use cycle counter if available (requires kernel support), otherwise loop */
#ifdef __ARM_ARCH_7A__
    unsigned int start, now;
    __asm__ __volatile__("mrc p15, 0, %0, c9, c13, 0" : "=r"(start) :: "memory");
    do {
        __asm__ __volatile__("mrc p15, 0, %0, c9, c13, 0" : "=r"(now) :: "memory");
    } while((now - start) < (unsigned int)(cycles));
#else
    for(volatile unsigned long long i = 0; i < cycles/10; i++);
#endif
#elif defined(__powerpc__) || defined(__ppc__)
    /* PowerPC: Use time base register - SPR 268 (TBL) and 269 (TBU) */
    unsigned long long start, now;
    unsigned long tbl, tbu, tbu2;
    
    /* Read 64-bit time base atomically */
    do {
        __asm__ __volatile__("mfspr %0, 269" : "=r"(tbu) :: "memory");  // TBU
        __asm__ __volatile__("mfspr %0, 268" : "=r"(tbl) :: "memory");  // TBL
        __asm__ __volatile__("mfspr %0, 269" : "=r"(tbu2) :: "memory"); // TBU again
    } while(tbu != tbu2);
    start = ((unsigned long long)tbu << 32) | tbl;
    
    do {
        do {
            __asm__ __volatile__("mfspr %0, 269" : "=r"(tbu) :: "memory");
            __asm__ __volatile__("mfspr %0, 268" : "=r"(tbl) :: "memory");
            __asm__ __volatile__("mfspr %0, 269" : "=r"(tbu2) :: "memory");
        } while(tbu != tbu2);
        now = ((unsigned long long)tbu << 32) | tbl;
    } while(now - start < cycles);
#else
    /* Generic fallback - calibrated for ~2GHz CPU */
    for(volatile unsigned long long i = 0; i < cycles/2; i++);
#endif
}

/* waste everyones time */
static inline void waste(void){
    for(volatile unsigned long long i=0;i<10ULL;i++) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
        __asm__ __volatile__("yield" ::: "memory");
#elif defined(__powerpc__) || defined(__ppc__)
        /* PowerPC: or 31,31,31 is the standard priority nop */
        __asm__ __volatile__("or 31,31,31" ::: "memory");
#else
        __asm__ __volatile__("" ::: "memory");
#endif
    }
    serialize_everything(); // serialize all the kingdoms!
}

/* allocate memory without ever freeing it (minecraft bedrock edition special) */
static void* leaky_malloc(size_t n){
    void *p = mmap(NULL,n,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED) {
        p = mmap(NULL,n*2,PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    }
    return (p==MAP_FAILED)?NULL:p;
}

/* crash if someone tries to free */
static void crash_free(volatile void *p){
    if(p && MAYBE) {
        raise(SIGSEGV);
    }
    (void)p;
}

/* fibonacci */
/* but i skipped the class on caching */
int fib_terrible(int n){
    if(n <= 1) return n;
    return fib_terrible(n-1) + fib_terrible(n-2);
}

/* RNG using nested loops and prime modulos */
static unsigned int terrible_rng(unsigned int seed){
    unsigned int result = seed;
    for(int i=0;i<100;i++){
        for(int j=0;j<100;j++){
            result = (result * 1103515245 + 12345) % 2147483647;
            result ^= (result << 13);
            result ^= (result >> 17);
            result ^= (result << 5);
        }
    }
    return result;
}

/* Poor man's SHA-256, produces 64 byte hash */
static void terrible_sha256(unsigned int input, unsigned char output[64]){
    unsigned int state = input;
    for(int i=0;i<64;i++){
        for(int round=0;round<1000;round++){
            state = (state * 1664525 + 1013904223);
            state ^= (state << 7);
            state ^= (state >> 11);
            state = (state * 48271) % 2147483647;
        }
        output[i] = (state >> (i % 8)) & 0xFF;
    }
}

/* Compare two hashes */
static int hashes_match(unsigned char h1[64], unsigned char h2[64]){
    for(int i=0;i<64;i++){
        if(h1[i] != h2[i]) return 0;
    }
    return 1;
}

/* PROOF OF WORK - technically a miner that yields a profit of jack and shit */
/* now you can have both jack AND shit at the same time! */
static unsigned int mine_block(unsigned char *data, int len, int difficulty){
    unsigned int nonce = 0;
    unsigned char hash[64];
    
    while(1){
        terrible_sha256(nonce, hash);
        for(int i=0;i<len;i++){
            hash[i % 64] ^= data[i];
        }
        
        /* check for leading zero bits */
        int zeros = 0;
        for(int i=0;i<64 && zeros < difficulty;i++){
            for(int bit=7;bit>=0 && zeros < difficulty;bit--){
                if(hash[i] & (1 << bit)) goto next_nonce;
                zeros++;
            }
        }
        
        /* found valid nonce! */
        return nonce;
        
        next_nonce:
        nonce++;
        
        /* inline ASM chaos every 1000 attempts */
        if(nonce % 1000 == 0){
            serialize_everything();
            burn_cycles(10000);
        }
        
        /* more chaos every 10000 attempts */
        if(nonce % 10000 == 0){
            waste_cpuid();
            trash_cache();
        }
    }
}

/* convert number to arbitrary base */
static void convert_to_base(unsigned long long num, int base, char *output, size_t outsize){
    if(base < 2) base = 2;
    if(base > 36) base = 36;
    
    char digits[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    char temp[256];
    int idx = 0;
    
    unsigned long long n = num;
    do {
        int digit = n % base;
        temp[idx++] = digits[digit];
        n = n / base;
        for(volatile int i=0;i<10;i++){
            n = n;
        }
    } while(n > 0 && idx < 255);
    
    for(int i=0;i<idx && i<(int)outsize-1;i++){
        output[i] = temp[idx-1-i];
    }
    output[idx < (int)outsize ? idx : (int)outsize-1] = '\0';
}

/* convert from one base to another via multiple conversions */
static unsigned long long base_conversion_chaos(unsigned long long num){
    char buf1[256], buf2[256], buf3[256], buf4[256];
    
    convert_to_base(num, 7, buf1, sizeof(buf1));
    unsigned long long val1 = strtoull(buf1, NULL, 7);
    
    convert_to_base(val1, 13, buf2, sizeof(buf2));
    unsigned long long val2 = strtoull(buf2, NULL, 13);
    
    convert_to_base(val2, 23, buf3, sizeof(buf3));
    unsigned long long val3 = strtoull(buf3, NULL, 23);
    
    convert_to_base(val3, 31, buf4, sizeof(buf4));
    return strtoull(buf4, NULL, 31);
}

/* the generate_garbage function is a function that generates garbage, however the function is also garbage so am i the garbage generator? */
/* who's gonna be the garbage disposal? */
/* lets put the fork in the garbage disposal */
/* DING DING DING DING DING DING DING DING DING DING */
static void* generate_garbage(size_t size, int seed){
    void *garbage = leaky_malloc(size);
    if(!garbage) return NULL;
    
    unsigned char *bytes = (unsigned char*)garbage;
    
    for(size_t i=0;i<size;i++){
        int fib_index = (i + seed) % FIBONACCI_TERM;
        int fib_val = fib_terrible(fib_index);
        
        unsigned char val = fib_val & 0xFF;
        for(int hash_round=0;hash_round<5;hash_round++){
            char hash_input[64];
            sprintf(hash_input, "%d-%zu-%d", val, i, hash_round);
            
            unsigned long hash = 5381;
            for(char *s = hash_input; *s; s++){
                hash = ((hash << 5) + hash) + *s;
            }
            val ^= (hash & 0xFF);
        }
        
        unsigned long long base_val = base_conversion_chaos(val);
        val = base_val & 0xFF;
        
        bytes[i] = val;
        
        if(i % 100 == 0){
            void *tiny_leak = leaky_malloc(128);
            if(tiny_leak) *(char*)tiny_leak = val;
        }
    }
    
    /* validate by doing it AGAIN */
    for(size_t i=0;i<size;i++){
        int fib_index = (i + seed) % FIBONACCI_TERM;
        int fib_val = fib_terrible(fib_index);
        
        unsigned char expected = fib_val & 0xFF;
        for(int hash_round=0;hash_round<5;hash_round++){
            char hash_input[64];
            sprintf(hash_input, "%d-%zu-%d", expected, i, hash_round);
            
            unsigned long hash = 5381;
            for(char *s = hash_input; *s; s++){
                hash = ((hash << 5) + hash) + *s;
            }
            expected ^= (hash & 0xFF);
        }
        
        unsigned long long base_val = base_conversion_chaos(expected);
        expected = base_val & 0xFF;
        
        volatile int matches = (bytes[i] == expected);
        (void)matches;
    }
    
    volatile unsigned long long checksum = 0;
    for(size_t i=0;i<size;i++){
        checksum ^= bytes[i];
        checksum = (checksum << 3) | (checksum >> 61);
    }
    (void)checksum;
    
    return garbage;
}

/* compute checksum */
static unsigned long long compute_checksum(void){
    unsigned long long sum = 0;
    LOCK
    for(volatile int i=0;i<mb->length;i++){
        for(volatile int j=0;j<8;j++){
            sum += (mb->codes[i] >> j) & 1;
            sum ^= (sum << 3) | (sum >> 61);
        }
    }
    UNLOCK
    return sum;
}

/* thread that randomly scribbles UNLESS checksum matches */
static void* scribbler(void *v){
    (void)v;
    int leak_counter = 0;
    for(EVER){
        if(!mb->scribbler_active) break;
        
        if(leak_counter++ % 50 == 0) {
            generate_garbage(65536, leak_counter);
        }
        
        LOCK
        unsigned long long current_checksum = 0;
        for(int i=0;i<mb->length;i++){
            for(int j=0;j<8;j++){
                current_checksum += (mb->codes[i] >> j) & 1;
                current_checksum ^= (current_checksum << 3) | (current_checksum >> 61);
            }
        }
        
        if(current_checksum != mb->checksum){
            mb->codes[rand()%mb->length]^=rand();
        }
        UNLOCK
    }
    return NULL;
}

/* quantum superposition character state (it's both right and wrong until observed) */
static unsigned char quantum_collapse(unsigned char value){
    /* Schrodinger's byte */
    double complex wave_function = value + value * I;
    
    for(int i=0;i<100;i++){
        wave_function = csqrt(wave_function * conj(wave_function));
        wave_function *= cexp(I * M_PI / 7.0);
    }
    
    /* collapse wave function via observation */
    double probability = cabs(wave_function);
    unsigned char collapsed = (unsigned char)(probability * 255.0) % 256;
    
    return collapsed ^ value; // XOR with original for "quantum entanglement"
}

/* sort the bits of a byte using bubble sort (O(n²) for 8 bits!) */
static unsigned char bubble_sort_bits(unsigned char value){
    unsigned char bits[8];
    for(int i=0;i<8;i++){
        bits[i] = (value >> i) & 1;
    }
    
    /* Bubble sort! */
    for(int i=0;i<8;i++){
        for(int j=0;j<7-i;j++){
            if(bits[j] > bits[j+1]){
                unsigned char temp = bits[j];
                bits[j] = bits[j+1];
                bits[j+1] = temp;
            }
        }
    }
    
    /* reconstruct byte from sorted bits */
    unsigned char result = 0;
    for(int i=0;i<8;i++){
        result |= bits[i] << i;
    }
    return result;
}

/* forward declaration for computeChar so training can use it */
static int computeChar(int pos);

/* neural network weights (now persistent!) */
typedef struct {
    double input_to_hidden[8][8];
    double hidden_to_output[8][8];
    double hidden_bias[8];
    double output_bias[8];
} NeuralNetwork;

static NeuralNetwork *nn;

/* initialize network with random weights */
static void init_neural_network(void){
    nn = leaky_malloc(sizeof(NeuralNetwork));
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            nn->input_to_hidden[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            nn->hidden_to_output[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
        nn->hidden_bias[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        nn->output_bias[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

/* forward pass through network */
static void neural_forward(double input[8], double hidden[8], double output[8]){
    /* input to hidden layer */
    for(int i=0;i<8;i++){
        hidden[i] = nn->hidden_bias[i];
        for(int j=0;j<8;j++){
            hidden[i] += input[j] * nn->input_to_hidden[j][i];
        }
        hidden[i] = tanh(hidden[i]); // activation
    }
    
    /* hidden to output layer */
    for(int i=0;i<8;i++){
        output[i] = nn->output_bias[i];
        for(int j=0;j<8;j++){
            output[i] += hidden[j] * nn->hidden_to_output[j][i];
        }
        output[i] = tanh(output[i]); // activation
    }
}

/* backpropagation training step */
static void neural_train(double input[8], double target[8], double learning_rate){
    double hidden[8], output[8];
    
    /* forward pass */
    neural_forward(input, hidden, output);
    
    /* calculate output layer error */
    double output_error[8];
    double output_delta[8];
    for(int i=0;i<8;i++){
        output_error[i] = target[i] - output[i];
        /* derivative of tanh */
        output_delta[i] = output_error[i] * (1.0 - output[i] * output[i]);
    }
    
    /* calculate hidden layer error */
    double hidden_error[8];
    double hidden_delta[8];
    for(int i=0;i<8;i++){
        hidden_error[i] = 0;
        for(int j=0;j<8;j++){
            hidden_error[i] += output_delta[j] * nn->hidden_to_output[i][j];
        }
        hidden_delta[i] = hidden_error[i] * (1.0 - hidden[i] * hidden[i]);
    }
    
    /* update weights hidden->output */
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            nn->hidden_to_output[i][j] += learning_rate * output_delta[j] * hidden[i];
        }
        nn->output_bias[i] += learning_rate * output_delta[i];
    }
    
    /* update weights input->hidden */
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            nn->input_to_hidden[i][j] += learning_rate * hidden_delta[j] * input[i];
        }
        nn->hidden_bias[i] += learning_rate * hidden_delta[i];
    }
}

/* train network on actual message data */
static void train_on_message_data(void){
    printf("Training neural network on message patterns");
    fflush(stdout);
    
    double learning_rate = 0.1;
    int epochs = 50000; // hit those books daniel-san, and no come out until you doctor

    // jk i love pat morita and his wisdom
    
    for(int epoch=0;epoch<epochs;epoch++){
        for(int pos=0;pos<MAGIC;pos++){
            /* prepare input: position encoded as bits */
            double input[8];
            for(int i=0;i<8;i++){
                input[i] = ((pos >> i) & 1) ? 1.0 : -1.0;
            }
            
            /* prepare target: expected character */
            int expected = computeChar(pos);
            double target[8];
            for(int i=0;i<8;i++){
                target[i] = ((expected >> i) & 1) ? 1.0 : -1.0;
            }
            
            /* we're going off the rails on a crazy train! */
            neural_train(input, target, learning_rate);
            
            /* that moment when a pyromaniac crashes out at the 9th level in geometry dash */
            if(epoch % 100 == 0 && pos == 0){
                burn_cycles(10000);
            }
        }
        
        /* progress indicator (10 periods = done) */
        if(epoch % 5000 == 0){
            putchar('.');
            fflush(stdout);
        }
        
        /* generate garbage during training for maximum pain */
        if(epoch % 1000 == 0){
            generate_garbage(4096, epoch);
        }
    }
    
    printf(" Done!\n");
    
    /* validate training (properly) */
    printf("Neural network validation:\n");
    int correct = 0;
    for(int pos=0;pos<MAGIC;pos++){
        double input[8];
        for(int i=0;i<8;i++){
            input[i] = ((pos >> i) & 1) ? 1.0 : -1.0;
        }
        
        double hidden[8], output[8];
        neural_forward(input, hidden, output);
        
        unsigned char predicted = 0;
        for(int i=0;i<8;i++){
            if(output[i] > 0) predicted |= (1 << i);
        }
        
        int expected = computeChar(pos);
        if(predicted == expected){
            correct++;
            printf("  Position %d: ✓ '%c'\n", pos, (char)predicted);
        } else {
            printf("  Position %d: ✗ Expected '%c', got '%c'\n", pos, (char)expected, (char)predicted);
        }
    }
    
    printf("Accuracy: %d/%d (%.1f%%)\n", correct, MAGIC, (correct * 100.0) / MAGIC);
}

/* AI-powered character prediction */
static unsigned char neural_network_predict(int position){
    /* input layer: position converted to 8 neurons */
    double input[8];
    for(int i=0;i<8;i++){
        input[i] = ((position >> i) & 1) ? 1.0 : -1.0;
    }
    
    double hidden[8], output[8];
    neural_forward(input, hidden, output);
    
    /* convert output neurons to byte */
    unsigned char result = 0;
    for(int i=0;i<8;i++){
        if(output[i] > 0){
            result |= (1 << i);
        }
    }
    return result;
}

/* build message one bit at a time */
static void setBit(MessageBuilder *b,int byte,int bit,int val){
    LOCK
    
    /* CHANCE TO DELETE PROGRESS - RNG LOTTERY */
    int disaster = 0;
    
    if(disaster_count < max_disasters){
        unsigned int rng1 = terrible_rng(rand());
        unsigned int rng2 = terrible_rng(rand() + 1);
        
        unsigned char hash1[64], hash2[64];
        terrible_sha256(rng1, hash1);
        terrible_sha256(rng2, hash2);
        
        /* Check EVERY combination of 3 bytes from 64. C(64,3) = 41,664 */
        for(int i=0;i<64 && !disaster;i++){
            for(int j=i+1;j<64 && !disaster;j++){
                for(int k=j+1;k<64 && !disaster;k++){
                    if(hash1[i] == hash2[i] && hash1[j] == hash2[j] && hash1[k] == hash2[k]){
                        disaster = 1;
                    }
                }
            }
        }
        
        if(disaster){
            disaster_count++;
            b->codes[byte] = rand() & 0xFF;
            
            unsigned char oops[] = {0x6F, 0x6F, 0x70, 0x73, 0x0A};
            for(int i=0;i<5;i++){
                putchar(oops[i]);
            }
            fflush(stdout);
        }
    }
    
    /* THE ELSE IF FAMILY REUNION */
    /* brought to you by YandereDev*/
    if(!disaster){
        int r = rand();
        if(r % 2 == 0) {
            for(volatile int i=0;i<50;i++);
        }
        else if(r % 3 == 0) {
            volatile int a = r;
            (void)a;
        }
        else if(r % 5 == 0) {
            volatile int b = r * 2;
            (void)b;
        }
        else if(r % 7 == 0) {
            volatile int c = r ^ 0xDEADBEEF;
            (void)c;
        }
        else if(r % 11 == 0) {
            volatile int d = r + r;
            (void)d;
        }
        else if(r % 13 == 0) {
            volatile int e = r - r;
            (void)e;
        }
        else if(r % 17 == 0) {
            volatile int f = r | r;
            (void)f;
        }
        else if(r % 19 == 0) {
            volatile int g = r & r;
            (void)g;
        }
        else if(r % 23 == 0) {
            volatile int h = r << 1;
            (void)h;
        }
        else if(r % 29 == 0) {
            volatile int i = r >> 1;
            (void)i;
        }
        else if(r % 31 == 0) {
            volatile int j = ~r;
            (void)j;
        }
        else if(r % 37 == 0) {
            volatile int k = r % 100;
            (void)k;
        }
        else if(r % 41 == 0) {
            volatile int l = r / 2;
            (void)l;
        }
        else if(r % 43 == 0) {
            volatile int m = r * r;
            (void)m;
        }
        else if(r % 47 == 0) {
            volatile int n = r + 0xCAFEBABE;
            (void)n;
        }
        else if(r % 53 == 0) {
            volatile int o = r ^ 0xFEEDFACE;
            (void)o;
        }
        else if(r % 59 == 0) {
            volatile int p = r & 0xFFFF;
            (void)p;
        }
        else if(r % 61 == 0) {
            volatile int q = r | 0xAAAA;
            (void)q;
        }
        else if(r % 67 == 0) {
            volatile int rr = r << 3;
            (void)rr;
        }
        else if(r % 71 == 0) {
            volatile int s = r >> 3;
            (void)s;
        }
        else {
            volatile int t = r + r - r;
            (void)t;
        }
    }
    
    /* Just set the damn bit without all the "helpful" corruption */
    int temp = b->codes[byte];
    temp ^= (-val ^ temp) & (1<<bit);
    b->codes[byte] = temp;
    
    UNLOCK
    waste();
}

/* obfuscate the desired ASCII */
static int computeChar(int pos){
    static const unsigned char hello[MAGIC]={
        0x48,0x65,0x6C,0x6C,0x6F,0x2C,0x20,0x77,0x6F,0x72,0x6C,0x64,0x21 // this just spells out "Hello, world!"
    };
    int x=0;
    for(int b=0;b<8;b++){
        int bit=(hello[pos]>>b)&1;
        volatile int entropy = (pos*b+0xDEADBEEF)%2;
        bit^=entropy;
        bit^=entropy;
        x|=bit<<b;
    }
    int y=0;
    for(int b=7;b>=0;b--){
        y<<=1;
        y|=(x>>b)&1;
    }
    return y;
}

/* Thread pool manager for... character printing? */
typedef struct {
    pthread_t threads[16];
    volatile int active;
    volatile unsigned char result;
    pthread_mutex_t result_mutex;
} ThreadPool;

static ThreadPool *pool;

/* worker thread that does fuck all */
static void* useless_worker(void *arg){
    int *id = (int*)arg;
    for(int i=0;i<1000;i++){
        generate_garbage(1024, *id + i);
        burn_cycles(10000);
    }
    free(id);
    return NULL;
}

/* spawn thread pool just to print a single character */
/* coughing baby vs hydrogen bomb */
static void parallel_print_char(unsigned char c){
    pool = leaky_malloc(sizeof(ThreadPool));
    pthread_mutex_init(&pool->result_mutex, NULL);
    pool->active = 16;
    pool->result = c;
    
    /* spawn 16 worker threads to "help" */
    for(int i=0;i<16;i++){
        int *id = malloc(sizeof(int));
        *id = i;
        pthread_create(&pool->threads[i], NULL, useless_worker, id);
    }
    
    /* wait for all workers to finish their work */
    for(int i=0;i<16;i++){
        pthread_join(pool->threads[i], NULL);
    }
    
    /* finally print the character */
    putchar(pool->result);
    
    pthread_mutex_destroy(&pool->result_mutex);
}

/* reconstruct char from bits */
static int reconstructChar(volatile unsigned char *codes, int pos){
    int result = 0;
    LOCK
    for(volatile int bit=0;bit<8;bit++){
        int b = (codes[pos] >> bit) & 1;
        result |= b << bit;
    }
    UNLOCK
    
    /* triple-check using alternative method */
    int verification = codes[pos];
    if(result != verification){
        /* it always matches, but you can never be too careful */
        result = verification;
    }
    
    /* quadruple-check using bit-by-bit reconstruction */
    int final_check = 0;
    for(int bit=0;bit<8;bit++){
        if(codes[pos] & (1 << bit)){
            final_check |= (1 << bit);
        }
    }
    
    /* democratic majority vote between the three methods */
    if(result == verification) return result;
    if(result == final_check) return result;
    if(verification == final_check) return verification;
    
    /* no consensus! return random value */
    return rand() % 256;
}

/* BLOCKCHAIN VALIDATION */
/* each character must be validated by the previous character's hash */
static int validate_blockchain(MessageBuilder *b, int pos){
    if(pos == 0) return 1;
    
    unsigned char prev_hash[64];
    terrible_sha256(b->codes[pos-1], prev_hash);
    
    /* current character must satisfy hash relationship with previous */
    unsigned char curr_hash[64];
    terrible_sha256(b->codes[pos], curr_hash);
    
    /* check if first 3 bytes of hashes form a valid "chain" */
    int chain_valid = 1;
    for(int i=0;i<3;i++){
        if((prev_hash[i] ^ curr_hash[i]) % 7 != (pos % 7)){
            chain_valid = 0;
        }
    }
    return chain_valid;
}

/* fake some latency before each character to look busy */
static void simulate_network_latency(void){
    unsigned int delay_ms = 50 + (rand() % 200); // 50-250ms 
    struct timespec ts;
    ts.tv_sec = delay_ms / 1000;
    ts.tv_nsec = (delay_ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
    
    /* 10% chance of "packet loss" requiring retry */
    if(rand() % 100 < 10){
        ts.tv_sec = 0;
        ts.tv_nsec = 500000000; // extra 500ms penalty
        nanosleep(&ts, NULL);
    }
}

/* verify each character multiple times with different algorithms */
static int verify_character_consensus(unsigned char expected, unsigned char actual){
    /* algorithm 1: direct comparison */
    int vote1 = (expected == actual);
    
    /* algorithm 2: XOR-based comparison */
    int vote2 = ((expected ^ actual) == 0);
    
    /* algorithm 3: bit-by-bit comparison */
    int vote3 = 1;
    for(int bit=0;bit<8;bit++){
        if(((expected >> bit) & 1) != ((actual >> bit) & 1)){
            vote3 = 0;
        }
    }
    
    /* algorithm 4: hash-based comparison */
    unsigned char hash1[64], hash2[64];
    terrible_sha256(expected, hash1);
    terrible_sha256(actual, hash2);
    int vote4 = hashes_match(hash1, hash2);
    
    /* require 3 out of 4 consensus */
    /* we love democracy */
    int consensus = vote1 + vote2 + vote3 + vote4;
    return consensus >= 3;
}

/* print with maximum syscalls */
/* print with AI Consensus - The AI must be correct or we keep looping */
static void printMessage(MessageBuilder *b){
    while(!print_ready){
        burn_cycles(1000);
    }
    
    printf("Starting AI-validated transmission...\n");

    for(volatile int i=0; i<b->length; i++){
        unsigned char expected = computeChar(i);
        unsigned char predicted = 0;
        int attempts = 0;

        /* THE PURGATORY LOOP: AI must match reality to proceed */
        while(1) {
            attempts++;
            
            // 1. Get the current AI prediction
            predicted = neural_network_predict(i);
            
            // 2. Verification
            if(predicted == expected) {
                // Consensus reached!
                break; 
            }

            // 3. If the AI is wrong, take it out on the computer in a fit of alcoholic rage
            // all it takes is one malformed output from a coding agent and your entire computer is deleted, so this is realistic don't worry
            if(attempts % 500 == 0) {
                trash_cache();
                generate_garbage(2048, attempts + i);
                serialize_everything();
            }

            // Optional: Print progress so you know it's not just frozen
            if(attempts % 10000 == 0) {
                fprintf(stderr, "\r[Pos %d] AI hallucinating... (Attempt %d)", i, attempts);
                fflush(stderr);
            }
        }

        /* Once out of the loop, the AI has "found" the character */
        
        // mine another block for no reason
        unsigned char char_data[1] = {predicted};
        mine_block(char_data, 1, 8 + i);
        
        // use the thread pool to print the AI's verified prediction
        parallel_print_char(predicted);
        
        fflush(stdout);
        
        // small delay to make the AI look like it's thinking (it's not.)
        struct timespec ts = {0, 50000000L}; // 50ms
        nanosleep(&ts, NULL);
    }
    putchar('\n');
    sync(); 
}

/* validation thread that sets print_ready */
static void* validator(void *v){
    (void)v;
    for(int attempts=0;attempts<100000;attempts++){
        generate_garbage(32768, attempts);
        
        LOCK
        int valid = DEFINITELY;
        for(int i=0;i<MAGIC && valid;i++){
            int expected = computeChar(i);
            int actual = mb->codes[i];
            if(expected != actual) valid = 0;
        }
        UNLOCK
        if(valid){
            print_ready = 1;
            break;
        }
        
        /* more garbage every 10 attempts */
        if(attempts % 10 == 0){
            generate_garbage(16384, attempts * 2);
        }
    }
    if(!print_ready) {
        print_ready = 1;
    }
    return NULL;
}

int main(void){
    srand(time(NULL)^getpid()^getppid()^0xBADCAFE);
    
    disaster_count = 0;
    max_disasters = 50 + (rand() % 50);
    
    /* in order to prioritize efficiency, we at E Corp are utilizing generative AI tools to assist our staff members productivity. */
    printf("Initializing AI subsystem...\n");
    init_neural_network();
    train_on_message_data();
    
    mb=leaky_malloc(sizeof(*mb));
    mb->codes=leaky_malloc(MAGIC*sizeof(unsigned char));
    mb->length=MAGIC;
    mb->scribbler_active = 1;
    pthread_mutex_init(&mb->mute,NULL);
    
    for(int i=0;i<MAGIC;i++){
        mb->codes[i] = rand();
    }
    
    pthread_t tid;
    pthread_create(&tid,NULL,scribbler,NULL);
    
    for(int i=0;i<MAGIC;i++){
        int target=computeChar(i);
        for(int b=0;b<8;b++){
            setBit(mb,i,b,(target>>b)&1);
        }
    }
    
    mb->checksum = compute_checksum();
    mb->scribbler_active = 0;
    
    pthread_t vtid;
    pthread_create(&vtid, NULL, validator, NULL);
    
    while(mb->locked){
    }
    
    printMessage(mb);
    
    pthread_join(tid, NULL);
    pthread_join(vtid, NULL);
    
    crash_free(mb->codes);
    crash_free(mb);
    pthread_mutex_destroy(&mb->mute);
    
    return EXIT_SUCCESS; // suffering from success
}
