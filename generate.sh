#!/bin/bash

python generate.py --model nanobanana2 --dim dimension_conflict --exp_name benchmark_v1
python generate.py --model nanobanana1 --dim dimension_conflict --exp_name benchmark_v1

python generate.py --model nanobanana2 --dim dimension_rule --exp_name benchmark_v1
python generate.py --model nanobanana1 --dim dimension_rule --exp_name benchmark_v1

python generate.py --model nanobanana2 --dim dimension_metaphor --exp_name benchmark_v1
python generate.py --model nanobanana1 --dim dimension_metaphor --exp_name benchmark_v1

python generate.py --model nanobanana2 --dim dimension_preference --exp_name benchmark_v1
python generate.py --model nanobanana1 --dim dimension_preference --exp_name benchmark_v1


python generate.py --model nanobanana2 --dim dimension_visual_link --exp_name benchmark_v1
python generate.py --model nanobanana1 --dim dimension_visual_link --exp_name benchmark_v1