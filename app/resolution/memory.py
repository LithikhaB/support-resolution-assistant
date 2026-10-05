"""Browser-owned redacted history; unresolved reports never become repair evidence."""

from contextvars import ContextVar
from uuid import uuid4

from psycopg.types.json import Jsonb

from app.config.settings import get_settings
from app.database.connection import get_connection
from app.llm.privacy import redact
from app.retrieval.models import EvidenceResult

history_context = ContextVar("history_context", default=None)


class ConversationConflict(ValueError):
    pass


class ConversationStore:
    def __init__(self, connection=get_connection, retention_days=None):
        self.connection = connection
        self.days = retention_days or get_settings().conversation_retention_days

    def load(self, owner, conversation_id):
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM support_conversations WHERE owner_hash=%s AND expires_at<=now()",
                (owner,),
            )
            row = conn.execute(
                "SELECT request,revision FROM support_conversations WHERE owner_hash=%s AND conversation_id=%s AND expires_at>now()",
                (owner, conversation_id),
            ).fetchone()
        return (
            {"request": row[0], "revision": row[1], "conversation_id": str(conversation_id)}
            if row
            else None
        )

    def recent(self, owner):
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM support_conversations WHERE owner_hash=%s AND expires_at<=now()",
                (owner,),
            )
            rows = conn.execute(
                "SELECT conversation_id,request,revision,updated_at FROM support_conversations WHERE owner_hash=%s AND expires_at>now() ORDER BY updated_at DESC LIMIT 20",
                (owner,),
            ).fetchall()
        return [
            {
                "conversation_id": str(r[0]),
                "query": r[1]["query"][:180],
                "revision": r[2],
                "updated_at": r[3].isoformat(),
            }
            for r in rows
        ]

    def save(self, owner, request, result, source_revision):
        cid = request.conversation_id or uuid4()
        masked_request, _ = redact(request.model_dump(mode="json"))
        masked_result, _ = redact(result.model_dump(mode="json"))
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM support_conversations WHERE owner_hash=%s AND expires_at<=now()",
                (owner,),
            )
            old = conn.execute(
                "SELECT revision,request FROM support_conversations WHERE conversation_id=%s AND owner_hash=%s FOR UPDATE",
                (cid, owner),
            ).fetchone()
            if request.conversation_id and (
                not old or request.revision != old[0] or masked_request["query"] != old[1]["query"]
            ):
                raise ConversationConflict("Conversation changed; reload it before replying.")
            revision = old[0] + 1 if old else 1
            masked_request.update(conversation_id=str(cid), revision=revision)
            if old:
                conn.execute(
                    "UPDATE support_conversations SET request=%s,revision=%s,updated_at=now(),expires_at=now()+%s*interval '1 day' WHERE conversation_id=%s AND owner_hash=%s",
                    (Jsonb(masked_request), revision, self.days, cid, owner),
                )
            else:
                conn.execute(
                    "INSERT INTO support_conversations(conversation_id,owner_hash,revision,request,expires_at) VALUES(%s,%s,%s,%s,now()+%s*interval '1 day')",
                    (cid, owner, revision, Jsonb(masked_request), self.days),
                )
            conn.execute(
                "INSERT INTO support_conversation_events VALUES(%s,%s,%s,%s)",
                (cid, revision, Jsonb(masked_result), Jsonb(list(source_revision))),
            )
            conn.execute(
                "DELETE FROM support_conversation_events WHERE conversation_id=%s AND revision<%s",
                (cid, revision - 19),
            )
            conn.execute(
                "DELETE FROM support_conversations WHERE conversation_id IN (SELECT conversation_id FROM support_conversations WHERE owner_hash=%s ORDER BY updated_at DESC OFFSET 100)",
                (owner,),
            )
        return str(cid), revision

    def delete(self, owner, cid):
        with self.connection() as conn:
            return bool(
                conn.execute(
                    "DELETE FROM support_conversations WHERE owner_hash=%s AND conversation_id=%s RETURNING conversation_id",
                    (owner, cid),
                ).fetchone()
            )

    def approve(self, owner, cid, issue_id, revision, outcome_note, source_revision, embedder):
        """Only explicit demo-operator review can promote a simulated resolution."""
        note, _ = redact({"text": outcome_note})
        with self.connection() as conn:
            current = conn.execute(
                "SELECT revision FROM support_conversations WHERE owner_hash=%s AND conversation_id=%s AND expires_at>now() FOR UPDATE",
                (owner, cid),
            ).fetchone()
            if not current or current[0] != revision:
                raise ConversationConflict(
                    "Conversation changed or is unavailable; reload before review."
                )
            events = conn.execute(
                "SELECT response FROM support_conversation_events WHERE conversation_id=%s AND source_revision=%s ORDER BY revision DESC",
                (cid, Jsonb(list(source_revision))),
            ).fetchall()
            issue = next(
                (
                    i
                    for e in events
                    for i in e[0]["issues"]
                    if i["issue_id"] == issue_id
                    and i["resolution"]["sources"]
                    and i["resolution"]["validation"]["status"] == "passed"
                ),
                None,
            )
            if issue is None:
                raise ConversationConflict(
                    "No currently valid sourced plan is available for this issue."
                )
            complaint = issue["analysis_text"]
            vector = embedder.embed_query(complaint)
            count = 0
            for source in issue["resolution"]["sources"]:
                steps = sorted(
                    (q for q in source["quotes"] if q["field"] == "plan_step"),
                    key=lambda q: q["step_id"],
                )
                if len(steps) != 5:
                    continue
                resolution = (
                    "Reviewed simulated outcome. These conditional steps do not confirm a new customer's cause.\nOutcome note: "
                    + note["text"]
                    + "\n"
                    + "\n".join(f"Step {q['step_id']}: {q['text']}" for q in steps)
                )
                conn.execute(
                    "INSERT INTO support_reviewed_resolutions(conversation_id,owner_hash,issue_id,source_id,source_revision,category,complaint,resolution,outcome_note,outcome_status,embedding) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'simulated_resolved',%s::vector) ON CONFLICT(conversation_id,issue_id,source_id) DO UPDATE SET source_revision=excluded.source_revision,category=excluded.category,complaint=excluded.complaint,resolution=excluded.resolution,outcome_note=excluded.outcome_note,embedding=excluded.embedding,reviewed_at=now()",
                    (
                        cid,
                        owner,
                        issue_id,
                        source["doc_id"],
                        Jsonb(list(source_revision)),
                        issue["resolution"]["analysis"]["category"],
                        complaint,
                        resolution,
                        note["text"],
                        str(vector),
                    ),
                )
                count += 1
        if not count:
            raise ConversationConflict("Only ordered, validated procedures can be reviewed.")
        return count

    def search_reviewed(self, owner, query, source_ids, source_revision, embedder):
        if source_ids == []:
            return []
        vector = str(embedder.embed_query(query))
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT h.history_id,h.conversation_id,h.issue_id,h.source_id,h.category,h.complaint,h.resolution,h.outcome_note FROM support_reviewed_resolutions h JOIN support_conversations c USING(conversation_id) WHERE h.owner_hash=%s AND c.owner_hash=%s AND c.expires_at>now() AND h.source_revision=%s AND (%s::text[] IS NULL OR h.source_id=ANY(%s)) AND h.embedding <=> %s::vector < %s ORDER BY h.embedding <=> %s::vector,h.reviewed_at DESC LIMIT 3",
                (
                    owner,
                    owner,
                    Jsonb(list(source_revision)),
                    source_ids,
                    source_ids,
                    vector,
                    0.1 if source_ids is None else 0.4,
                    vector,
                ),
            ).fetchall()
        return [
            EvidenceResult(
                chunk_id=r[0],
                doc_id=f"recent_{r[1]}_{r[2]}_{r[3]}",
                chunk_index=0,
                title="Reviewed simulated conversation",
                doc_type="resolved_ticket",
                content=r[5],
                response=None,
                resolution=r[6],
                outcome_status="simulated_resolved",
                metadata={
                    "is_synthetic": True,
                    "category": r[4],
                    "kb_refs": [r[3]],
                    "outcome_evidence": r[7],
                    "origin": "browser_owned_reviewed_conversation",
                },
            )
            for r in rows
        ]
