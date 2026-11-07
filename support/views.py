
import json
import logging

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_POST, require_GET

from orders.models import Order
from .gemini_agents import run_support_agent
from .event_queue import publish, subscribe, unsubscribe
from .models import Conversation, Message


logger = logging.getLogger(__name__)


@login_required
@require_POST
def chat(request, order_id):

    # Accept only JSON requests.
    if not request.content_type or request.content_type != "application/json":
        return JsonResponse(
            {"error": "Content-Type must be application/json."},
            status=415
        )

    # Limit request size to 16 KB.
    if len(request.body) > 16 * 1024:
        return JsonResponse(
            {"error": "Request is too large."},
            status=413
        )

    # Validate JSON.
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {"error": "Invalid JSON request."},
            status=400
        )

    if not isinstance(data, dict):
        return JsonResponse(
            {"error": "Invalid request format."},
            status=400
        )

    user_message = data.get("message")

    if not isinstance(user_message, str):
        return JsonResponse(
            {"error": "Message must be text."},
            status=400
        )

    user_message = user_message.strip()

    if not user_message:
        return JsonResponse(
            {"error": "Message cannot be empty."},
            status=400
        )

    if len(user_message) > 4000:
        return JsonResponse(
            {"error": "Message cannot exceed 4000 characters."},
            status=400
        )

    # Only allow customers to access their own orders.
    order = get_object_or_404(
        Order,
        id=order_id,
        user=request.user
    )

    conversation, created = Conversation.objects.get_or_create(
        user=request.user,
        order=order
    )

    Message.objects.create(
        conversation=conversation,
        role="user",
        content=user_message
    )

    publish(conversation.id, {
        "type": "user_message",
        "message": user_message,
        "name": request.user.first_name
    })

    # Run the AI agent safely.
    try:
        reply = run_support_agent(
            user_message,
            conversation.id,
            order.id,
            request.user.id
        )

        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("AI returned an empty response.")

        Message.objects.create(
            conversation=conversation,
            role="assistant",
            content=reply
        )

        return JsonResponse({"reply": reply})

    except Exception:
        logger.exception(
            "AI support processing failed for conversation %s",
            conversation.id
        )

        publish(conversation.id, {
            "type": "error",
            "message": "The AI assistant is temporarily unavailable."
        })

        return JsonResponse(
            {
                "error": "The AI assistant is temporarily unavailable. Please try again."
            },
            status=502
        )


@staff_member_required
def dashboard(request):

    conversations = Conversation.objects.all().order_by("-created_at")

    return render(
        request,
        "support/dashboard.html",
        {"conversations": conversations}
    )


@staff_member_required
def conversation_detail(request, conversation_id):

    conversation = get_object_or_404(
        Conversation,
        id=conversation_id
    )

    messages = conversation.messages.order_by("created_at")
    agentlogs = conversation.agentlogs.order_by("created_at")

    context = {
        "conversation": conversation,
        "messages": messages,
        "agentlogs": agentlogs
    }

    return render(
        request,
        "support/conversation_detail.html",
        context
    )


@staff_member_required
@require_GET
def conversation_stream(request, conversation_id):

    # Confirm that the requested conversation exists.
    conversation = get_object_or_404(
        Conversation,
        id=conversation_id
    )

    def event_stream():
        q = subscribe(conversation.id)

        try:
            while True:
                event = q.get()

                yield f"data: {json.dumps(event)}\n\n"

        finally:
            unsubscribe(conversation.id, q)

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream"
    )

    response["Cache-Control"] = "no-cache, no-store"
    response["X-Accel-Buffering"] = "no"
    response["X-Content-Type-Options"] = "nosniff"

    return response
