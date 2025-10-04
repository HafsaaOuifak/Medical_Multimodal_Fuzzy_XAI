Step 0: One-time Preparation

Ensure Docker Desktop is running

Check that your project folder is shared in Docker Desktop
(Preferences → Resources → File Sharing)

Delete any empty local result folders (do this every time before a new run):

rm -rf tabular_models image_models multimodal_results surrogates

Step 1: Build and Start Docker Containers
docker compose build
docker compose up -d

Step 2: Mount Google Drive in the Container

Open a new terminal, keep this running while you work:

docker exec -it mmp-python bash -lc "bash /workspace/scripts/mount_gdrive.sh"

Step 3: Sanity Test the Volume Mount

Try this inside the running container:

docker exec -it mmp-python bash -lc "echo mounttest > /workspace/tabular_models/hostmount_test.txt"
cat tabular_models/hostmount_test.txt


If you see mounttest, the mount works. If not, follow the previous troubleshooting (share folder, restart Docker Desktop, etc).

Step 4: Train Tabular Models
docker exec -it mmp-python bash -lc "python -m src.python.train_tabular"


Check outputs:

ls tabular_models
ls tabular_models/DecisionTree

Step 5: Train Image Models
docker exec -it mmp-python bash -lc "python -m src.python.train_image"


Check outputs:

ls image_models
ls image_models/ResNet50

Step 6: Fuse Multimodal Results
docker exec -it mmp-python bash -lc "python -m src.python.fuse_multimodal"


Check outputs:

ls multimodal_results
cat multimodal_results/fused_metrics.csv


How to use the new option

Explain with all features (default):

docker exec -it mmp-python bash -lc "python -m src.python.explain --id 23"


Explain using only the top-10 most important surrogate features:

docker exec -it mmp-python bash -lc "python -m src.python.explain --id 23 --top_k 10"


Run comparison:

docker exec -it mmp-python bash -lc "python -m src.python.compare_xai --id 23"