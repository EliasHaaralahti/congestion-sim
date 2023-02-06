from data_models.output_summary import OutputSummary


def processor(env, detection_data: list[OutputSummary]):
    while True:
        print(f"Processing data of length {len(detection_data)}")
        # TODO: Implement some basic matching for YOLO results.
        yield env.timeout(1)
        