"""Download and rebuild the public census, weather and REPOS inputs."""
from wlpr.io import paths
from wlpr.io.public_inputs import main


if __name__ == "__main__":
    paths.assert_not_source_tree()
    main()
