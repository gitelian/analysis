



# 2133 first experiment, 794 trials, dynamice PC%, Good->Great
# 2139 2.1mW, 1002 trials, GREAT
# 2144 2.1mW, DISENGAGED entire experiment
# 2146 2.1mW, 604 trials, GREAT

# 1.1mW exps
# 2164, 65, 66 all great!

experiments = {
        'S8':
        {'JB116_R':
            [(2133,'TODO'),
                (2138, '2.2mW vM1, great experiment, 639 trials'),
                (2139,'2.2mW vM1, amazing, 1002 trials, great example experiment'),
                (2140,'trial 241 (1-based) dropped frames, mostly disengaged, 289 trials'),
                (2144,'2.2mW vM1, great, smashes whiskers into object, totally disengaged but clear retraction with light'),
                (2146, '2.2mW vM1, great, 604 trials')],
        'GT032_CE':
            [(2154, '2.2mW vM1, consistent, 1st day upstairs'),
                (2158, '2.2mW vM1, great'),
                (2161, '2.2mW vM1, good'),
                (2164, '1.1mW vM1, amazing'),
                (2167, '0.55mW vM1, subtle change'),
                (2170, '7.0mW vS1, dramatic, makes mouse sluggish'),
                (2172, '2.3mW vS1, good, slower performance'),
                (2179, '5.0mW, great, lots of trials'),
                (2183, '0.55mW vM1, lots of trials, no change?')],
        'GT027_RL':
            [(2155, '2.2mW vM1, great, 110min 1st day upstairs'),
                (2160, '2.2mW vM1, mix of performance'),
                (2162, '2.2mW vM1, great'),
                (2165, '1.1mW vM1, great, licks earlier with light'),
                (2168, '0.5mW vM1, great, 10% diff instead of 20%, manually had to overwrite hsv_times for 1 trial'),
                (2171, '7.0mW vS1, 250 trials in 80min, HSV problems, manually had to overwrite some data in hsv_times'),
                (2173, '2.3mW vS1, great but slow, licks later with light, had to manually overwrite 1 hsv_times trial'),
                (2176, '0.25mW vM1, great! lots of trials, slight decrease?'),
                (2180, '5.0mW vM1, great, had to manually overwrite 5 hsv_times trials'),
                (2182, '1.1mW vM1, great, motivated, lots of trials, had to manually overwrite 2 hsv_times trials'),
                (2184, '0.55mW vM1, great, motivated, lots of trials'),
                (2186, '3.0mW +1.1mm aterior to vM1, mixed performance'),
                (2189, '1.1mW +1.1mm anterior to vM1, great performance, no change with light??'),
                (2191, '1.1mW vM1, great performance,fiber may have been slightly too close to skull and in wrong spot??? idk why performance didnt drop more')],
        'GT027_RT':
            [(2157, '2.2mW vM1, great but only 202 trials, 1st day upstairs'),
                (2159, '2.2mW vM1, moderate trials'),
                (2163, '2.2mW vM1, great, lots of running, mix of correct/incorrect, disengaged last 10-15min, had to manually overwrite 1 hsv_times trial'),
                (2166, '1.1mW vM1, great, mix of good and poor performane'),
                (2169, '0.55mW vM1, okay, drop in performance around 40min'),
                (2174, '2.3mW vS1, okay, few motivated trials, vS1 light made mouse act weird'),
                ('NOTE', 'SLOW RUNSPEED used to filter trials')]},
        'S1/A1':
        {'GT027_NT':
            [(2149, 'mouse does better with vM1 light (~3mW), pole just out of reach of whiskers'),
                (2150, 'great performance, lots of running and licking, no difference with light, drop in performance towards the end'),
                (2152, 'first 35min great performance, last 15min drop in performance')],
        'GT027_CE':
            [(2151, 'great, 325 trials, mixed performance, mostly good'),
                (2153, 'first 15min water valve off, lots of running, no licking, performance increase to 100% after 25min')]
                }}




