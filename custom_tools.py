
import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")
ADZUNA_LOCATION = os.getenv("ADZUNA_LOCATION", "")

# Load course vector database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


@tool
def search_course_material(question: str) -> str:
    """Search ONLY the student's saved course PDF for concepts,
    definitions and explanations. Do not use for job searches."""

    documents = retriever.invoke(question)

    if not documents:
        return "No relevant information found in the course material."

    return "\n\n".join(doc.page_content for doc in documents)


@tool
def search_jobs_internships(topic: str) -> str:
    """Search actual external Adzuna job listings for jobs and
    internships related to a skill or role. For student internship
    searches, prioritize internships and entry-level roles, exclude
    clearly senior positions, and report only details returned by
    the API. Never invent companies, requirements or links."""

    topic = topic.strip()

    if not topic:
        return "Please provide a skill or job role to search for."

    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return "Adzuna credentials are missing from the .env file."

    url = (
        f"https://api.adzuna.com/v1/api/jobs/"
        f"{ADZUNA_COUNTRY}/search/1"
    )

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": topic,
        "results_per_page": 20,
        "content-type": "application/json"
    }

    if ADZUNA_LOCATION:
        params["where"] = ADZUNA_LOCATION

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        return "Adzuna request timed out. Please try again."

    except requests.exceptions.HTTPError as error:
        return f"Adzuna HTTP error: {error}"

    except requests.exceptions.RequestException as error:
        return f"Could not connect to Adzuna: {error}"

    except ValueError:
        return "Adzuna returned invalid JSON."

    jobs = data.get("results", [])

    if not jobs:
        return f"No Adzuna listings were returned for '{topic}'."

    topic_lower = topic.lower()

    internship_search = any(term in topic_lower for term in (
        "intern", "entry-level", "entry level",
        "fresher", "graduate"
    ))

    # Exclude clearly senior roles from student-level results.
    senior_terms = (
        "senior", "principal", "staff engineer",
        "architect", "director", "vice president",
        "head of ", "team lead", "engineering manager",
        "manager"
    )

    internship_terms = (
        "intern", "internship", "trainee",
        "entry-level", "entry level", "graduate",
        "fresher", "early career"
    )

    # Ignore generic search words when checking skill relevance.
    ignored_words = {
        "intern", "internship", "entry", "level",
        "job", "jobs", "role", "roles", "for",
        "and", "the", "a", "an", "student",
        "students", "graduate", "fresher",
        "opportunity", "opportunities"
    }

    topic_words = {
        word.strip(".,!?():;").lower()
        for word in topic.split()
        if len(word.strip(".,!?():;")) > 1
        and word.strip(".,!?():;").lower() not in ignored_words
    }

    eligible_jobs = []

    for job in jobs:
        title = job.get("title") or "Title not provided"
        description = job.get("description") or ""
        title_lower = title.lower()
        searchable_text = f"{title} {description}".lower()

        # Skip clearly senior positions for student searches.
        if internship_search and any(
            term in title_lower for term in senior_terms
        ):
            continue

        # Prioritize actual internship/entry-level listings.
        is_student_role = any(
            term in searchable_text for term in internship_terms
        )

        if internship_search and not is_student_role:
            continue

        matched_words = sorted(
            word for word in topic_words
            if word in searchable_text
        )

        # Require some evidence of a topic match.
        if topic_words and not matched_words:
            continue

        eligible_jobs.append((job, matched_words))

    if not eligible_jobs:
        return (
            f"No listings matched '{topic}' after filtering. "
            "Try a broader search such as 'Python Internship' "
            "or 'Machine Learning Intern'."
        )

    output = [
        f"Adzuna results for: {topic}",
        f"Matching listings: {min(len(eligible_jobs), 5)}",
        "Only details supplied by Adzuna are reported."
    ]

    for index, (job, matched_words) in enumerate(
        eligible_jobs[:5], start=1
    ):
        title = job.get("title") or "Title not provided"

        company_data = job.get("company") or {}
        company = company_data.get("display_name", "Not provided")

        location_data = job.get("location") or {}
        location = location_data.get("display_name", "Not provided")

        description = job.get("description") or ""
        application_url = job.get("redirect_url") or "Not provided"
        created = job.get("created") or "Not provided"

        output.append(
            f"\n{index}. {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Posted: {created}\n"
            f"Matched search terms: "
            f"{', '.join(matched_words) or 'None confirmed'}\n"
            f"Description from listing: {description[:500]}\n"
            f"Application link: {application_url}"
        )

    return "\n".join(output)


if __name__ == "__main__":
    print("Course tool:", search_course_material.name)
    print("Opportunity tool:", search_jobs_internships.name)

    print("\nTesting internship search...")
    print(search_jobs_internships.invoke({
    "topic": "Machine Learning Internship"
}))