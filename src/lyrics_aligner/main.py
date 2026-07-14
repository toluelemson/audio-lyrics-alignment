import logging

from lyrics_aligner.config import AppConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = AppConfig()
    logging.getLogger(__name__).info(
        "Audio-to-lyrics alignment foundation ready sample_rate=%s block_size=%s",
        config.sample_rate,
        config.block_size,
    )


if __name__ == "__main__":
    main()
