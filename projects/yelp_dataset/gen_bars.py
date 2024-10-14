import os
import json


def read_lines_in_batches(file_path, batch_size=100):
    with open(file_path, "r") as file:
        batch = []
        for line in file:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def process_json_flat_file(file_path, callback, batch_size=1000):
    results = []
    for batch in read_lines_in_batches(file_path, batch_size):
        batch_results = callback(batch)
        results.extend(batch_results)
    return results


CATEGORIES = [
    "Pubs",
    "Nightlife",
    "Food",
    "Bars",
    "Restaurants",
    "Hotels & Travel",
]

STATES = ["PA"]
CITIES = ["Philadelphia"]

BUSINESS_FIELDS = [
    "business_id",
    "name",
    "address",
    "city",
    "state",
    "postal_code",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "categories",
]

REVIEW_FIELDS = [
    "business_id",
    "stars",
    "text",
    "date",
]


def load_flat_json_batch_string(s):
    batch_json = "[" + ",".join(s) + "]"
    return json.loads(batch_json)


def filter_businesses(batch):
    businesses = load_flat_json_batch_string(batch)
    filtered = []
    for b in businesses:
        if b["state"] not in STATES:
            continue
        if b["city"] not in CITIES:
            continue
        if not b["categories"]:
            continue
        found = False
        for cat in b["categories"].split(", "):
            if cat in CATEGORIES:
                found = True
                break
        if not found:
            continue
        attrs = b["attributes"]
        if not attrs:
            continue
        alc = attrs.get("Alcohol", None)
        if alc in ["'none'", "u'none'", None]:
            continue
        filtered_business = {k: b[k] for k in BUSINESS_FIELDS}
        filtered.append(filtered_business)
    return filtered


def gen_filter_reviews(filter_ids):
    def filter_reviews(batch):
        reviews = load_flat_json_batch_string(batch)
        filtered = []
        for r in reviews:
            if r["business_id"] not in filter_ids:
                continue
            filtered_review = {k: r[k] for k in REVIEW_FIELDS}
            filtered.append(filtered_review)
        return filtered

    return filter_reviews


def main():
    pwd = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(pwd, "localdata")

    businesses = process_json_flat_file(
        f"{data_dir}/yelp_academic_dataset_business.json",
        filter_businesses,
    )
    business_map = {b["business_id"]: b for b in businesses}
    business_ids = set(business_map.keys())
    reviews = process_json_flat_file(
        f"{data_dir}/yelp_academic_dataset_review.json",
        gen_filter_reviews(business_ids),
    )
    for r in reviews:
        business = business_map[r["business_id"]]
        if "reviews" not in business:
            business["reviews"] = []
        business["reviews"].append(r)

    with open(f"{data_dir}/PA_Philadelphia_bars.json", "w") as f:
        json.dump(businesses, f, indent=2)


if __name__ == "__main__":
    main()
