uv run run_yolo.py \
        --data ./dataset/data.yaml \
        --model yolo11n.pt \
        --epochs 100 \
        --batch -1 \
        --imgsz 640 \
        --device 0 \
        --do_val \
        --tag baseline