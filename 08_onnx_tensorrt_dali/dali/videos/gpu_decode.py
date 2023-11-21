from typing import List

import cv2
import pandas as pd
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class VideoPipe(Pipeline):
    def __init__(
        self,
        batch_size: int,
        num_threads: int,
        device_id: int,
        filenames: List[str],
        sequence_length: int,
        step: int,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=12)
        self.filenames = filenames
        self.sequence_length = sequence_length
        self.step = step

    def define_graph(self):
        frames = fn.readers.video(
            device="cpu",
            filenames=self.filenames,
            sequence_length=self.sequence_length,
            normalized=False,
            random_shuffle=False,
            image_type=types.RGB,
            dtype=types.UINT8,
            step=self.step,
            file_list_include_preceding_frame=False,
        )
        return frames


def process_video(video_path: str) -> None:
    # Get FPS and total frames of video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    sequence_length = int(total_frames / fps)

    # Define and build pipe
    pipe = VideoPipe(
        batch_size=1,
        num_threads=8,
        device_id=3,
        filenames=[video_path],
        sequence_length=sequence_length,
        step=int(fps),
    )
    pipe.build()

    # Run pipe
    pipe_out = pipe.run()
    out = pipe_out[0].as_cpu().as_array()
    print(out.shape)


def main():
    for video_path in pd.read_parquet("video_paths.parquet").values:
        process_video(video_path[0])


if __name__ == "__main__":
    main()
