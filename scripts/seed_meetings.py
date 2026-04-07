"""Seed script to generate and index 500+ synthetic meeting notes."""

import asyncio
import random

from faker import Faker
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.db import Meeting
from app.services.embeddings import embedding_service
from app.utils.logger import get_logger

logger = get_logger(__name__)
fake = Faker()


MEETING_TYPES = [
    "Sprint Planning",
    "Sprint Retrospective",
    "Design Review",
    "Stakeholder Update",
    "1:1 Meeting",
    "Team Sync",
    "Product Review",
    "Technical Discussion",
    "Quarterly Planning",
    "Architecture Review",
]


async def generate_meeting_transcript(meeting_type: str) -> str:
    """Generate a realistic meeting transcript."""
    attendees = [fake.name() for _ in range(random.randint(3, 8))]
    
    transcript_parts = [
        f"Meeting Type: {meeting_type}",
        f"Attendees: {', '.join(attendees)}",
        f"Date: {fake.date_this_year()}",
        "",
    ]
    
    num_topics = random.randint(3, 6)
    for _ in range(num_topics):
        speaker = random.choice(attendees)
        topic = fake.sentence(nb_words=8)
        discussion = fake.paragraph(nb_sentences=random.randint(3, 6))
        transcript_parts.append(f"{speaker}: {topic}")
        transcript_parts.append(discussion)
        transcript_parts.append("")
    
    return "\n".join(transcript_parts)


async def generate_summary(transcript: str, meeting_type: str) -> str:
    """Generate a meeting summary."""
    summaries = [
        f"The team discussed key priorities for the upcoming sprint in this {meeting_type}.",
        f"This {meeting_type} covered progress updates and identified several blockers.",
        f"The {meeting_type} focused on aligning stakeholders on the product roadmap.",
        f"Team members shared updates and discussed technical challenges in this {meeting_type}.",
        f"The {meeting_type} reviewed recent accomplishments and planned next steps.",
    ]
    
    return random.choice(summaries) + " " + fake.sentence(nb_words=15)


async def generate_action_items() -> list[dict]:
    """Generate action items."""
    num_items = random.randint(2, 5)
    items = []
    
    for _ in range(num_items):
        items.append({
            "owner": fake.name(),
            "task": fake.sentence(nb_words=8).rstrip('.'),
            "deadline": fake.date_between(start_date='today', end_date='+30d').isoformat() if random.random() > 0.3 else None,
        })
    
    return items


async def generate_decisions() -> list[dict]:
    """Generate decisions."""
    num_decisions = random.randint(1, 3)
    decisions = []
    
    decision_templates = [
        "Approved the proposal to",
        "Decided to postpone",
        "Agreed to proceed with",
        "Rejected the idea of",
        "Committed to implementing",
    ]
    
    for _ in range(num_decisions):
        decisions.append({
            "decision": f"{random.choice(decision_templates)} {fake.sentence(nb_words=6).rstrip('.')}",
            "context": fake.sentence(nb_words=10) if random.random() > 0.5 else None,
        })
    
    return decisions


async def check_existing_seed_data() -> int:
    """Check if seed data already exists."""
    async with AsyncSessionLocal() as session:
        stmt = select(Meeting)
        result = await session.execute(stmt)
        meetings = result.scalars().all()
        return len(meetings)


async def seed_meetings(count: int = 500) -> None:
    """Seed the database with synthetic meeting data."""
    logger.info(f"Starting to seed {count} meetings")
    
    existing_count = await check_existing_seed_data()
    if existing_count >= count:
        logger.info(f"Database already has {existing_count} meetings. Skipping seed.")
        return
    
    logger.info(f"Found {existing_count} existing meetings. Seeding {count - existing_count} more.")
    
    async with AsyncSessionLocal() as session:
        for i in range(count - existing_count):
            meeting_type = random.choice(MEETING_TYPES)
            
            title = f"{meeting_type} - {fake.catch_phrase()}"
            transcript = await generate_meeting_transcript(meeting_type)
            summary = await generate_summary(transcript, meeting_type)
            action_items = await generate_action_items()
            decisions = await generate_decisions()
            
            content = f"{title}\n\n{summary}\n\n{transcript}"
            embedding = await embedding_service.generate_embedding(content)
            
            meeting = Meeting(
                title=title,
                transcript=transcript,
                summary=summary,
                action_items=action_items,
                decisions=decisions,
                confidence=random.uniform(0.75, 1.0),
                status="confirmed",
                embedding=embedding,
            )
            
            session.add(meeting)
            
            if (i + 1) % 50 == 0:
                await session.commit()
                logger.info(f"Seeded {i + 1}/{count - existing_count} meetings")
        
        await session.commit()
    
    logger.info(f"Successfully seeded {count - existing_count} meetings")


async def main() -> None:
    """Main entry point for the seed script."""
    try:
        await seed_meetings(500)
        logger.info("Seed script completed successfully")
    except Exception as e:
        logger.error(f"Seed script failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
