"""Delete all archived tasks and their artifacts from ClearML."""

from clearml import Task
from clearml.backend_api.session.client import APIClient


def main():
    client = APIClient()

    all_task_ids = []
    page = 0
    while True:
        resp = client.tasks.get_all(
            search_hidden=True,
            only_fields=["id", "system_tags"],
            page_size=500,
            page=page,
        )
        if not resp:
            break
        for t in resp:
            if t.system_tags and "__$ans_archived" in t.system_tags:
                all_task_ids.append(t.id)
        page += 1

    print(f"Found {len(all_task_ids)} archived tasks")
    if not all_task_ids:
        return

    for i, tid in enumerate(all_task_ids, 1):
        print(f"[{i}/{len(all_task_ids)}] Deleting {tid}")
        Task.delete(task=tid, delete_artifacts_and_models=True)

    print("Done")


if __name__ == "__main__":
    main()
