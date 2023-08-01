zip_name="default"

zip submissionB/${zip_name}.zip test_predictions.json post_generate_process.py
mv test_predictions.json resultsB/$zip_name.json

