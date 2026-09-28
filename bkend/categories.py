"""
V3.1 - ADDED SECTION BASED - section taxonomy, WHY?
each section:
  id       - frontend when student picks a button
  label    - show to frontend
  exemplars - handful representativ query used embed a
              centroid for intent routing when no section passed in
  urlhints - substrings expect in source urls use to
             boost retrieval scoring for chunks from matching pages
"""
sections = [
    {
        "id": "general",
        "label": "General Helpdesk",
        "exemplars": [
            "what can you help me with",
            "how do i contact the department",
            "where is the office",
            "office timings",
        ],
        "urlhints": ["show-contactus", "index.php"],
    },
    {
        "id": "admissions",
        "label": "Admissions",
        "exemplars": [
            "how do i apply for admission",
            "eligibility criteria for mca",
            "admission fee and deadline",
            "entrance test pu cet",
            "merit list reservation",
        ],
        "urlhints": ["nbid=4", "admission"],
    },
    {
        "id": "academics",
        "label": "Academics & Courses",
        "exemplars": [
            "what courses does dcsa offer",
            "syllabus for mca semester 1",
            "course structure and credits",
            "who teaches this subject",
            "attendance requirement",
        ],
        "urlhints": ["show-courses", "nbid=13", "syllab"],
    },
    {
        "id": "examinations",
        "label": "Examinations",
        "exemplars": [
            "exam date sheet",
            "admit card download",
            "revaluation process",
            "supplementary exam",
            "result declaration date",
        ],
        "urlhints": ["exam", "nbid=1"],
    },
    {
        "id": "fees",
        "label": "Fees & Payments",
        "exemplars": [
            "semester fee amount",
            "how to pay fees online",
            "fee refund policy",
            "late fee deadline",
        ],
        "urlhints": ["fee"],
    },
    {
        "id": "facultycontacts",
        "label": "Faculty, Staff & Contacts",
        "exemplars": [
            "who is the chairperson",
            "faculty list computer science",
            "department phone number and email",
            "office address",
        ],
        "urlhints": ["show-faculty", "show-contactus"],
    },
    {
        "id": "notices",
        "label": "Notices & Academic Calendar",
        "exemplars": [
            "latest academic calendar",
            "recent notices",
            "semester start date",
            "revised date sheet notification",
        ],
        "urlhints": ["show-noticeboard"],
    },
    {
        "id": "facilities",
        "label": "Facilities & Campus",
        "exemplars": [
            "library timings",
            "wifi registration",
            "computer lab access",
            "hostel accommodation",
        ],
        "urlhints": ["facilities"],
    },
    {
        "id": "placements",
        "label": "Placements & Internships",
        "exemplars": [
            "placement assistance",
            "internship registration",
            "tpc contact",
            "eligibility for campus placement",
        ],
        "urlhints": ["nbid=47", "placement"],
    },
    {
        "id": "research",
        "label": "Research & PhD",
        "exemplars": [
            "phd admission eligibility",
            "research scholar list",
            "phd supervisor",
            "thesis submission",
        ],
        "urlhints": ["show-researchscholar", "show-theses"],
    },
]
sectionbyid = {s["id"]: s for s in sections}
