from django.utils import timezone
from decimal import Decimal
import json

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from orders.models import Order
from support.models import Conversation


class SupportSecurityTests(TestCase):

    def setUp(self):
        # Create two normal customers.
        self.customer = User.objects.create_user(
            username="customer1",
            password="TestPassword123!"
        )

        self.other_customer = User.objects.create_user(
            username="customer2",
            password="TestPassword123!"
        )

        # Create a staff user.
        self.staff_user = User.objects.create_user(
            username="staff1",
            password="TestPassword123!",
            is_staff=True
        )

        # Create an order belonging to customer1.
        self.order = Order.objects.create(
            user=self.customer,
            product_name="CoolBreeze Smart AC",
            amount=Decimal("35000.00"),
            status="dispatched",
            carrier="BlueDart",
            tracking_number="CBTEST12345",
            delivery="Hyderabad, Telangana",
            created_at=timezone.now()
)

        # Create a conversation for customer1.
        self.conversation = Conversation.objects.create(
            user=self.customer,
            order=self.order
        )

    def test_anonymous_user_cannot_access_chat(self):
        response = self.client.post(
            reverse("chat", args=[self.order.id]),
            data=json.dumps({"message": "Hello"}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 302)

    def test_customer_cannot_access_another_customers_order(self):
        self.client.login(
            username="customer2",
            password="TestPassword123!"
        )

        response = self.client.post(
            reverse("chat", args=[self.order.id]),
            data=json.dumps({"message": "Show my order"}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 404)

    def test_customer_cannot_access_dashboard(self):
        self.client.login(
            username="customer1",
            password="TestPassword123!"
        )

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 302)

    def test_customer_cannot_access_conversation_detail(self):
        self.client.login(
            username="customer1",
            password="TestPassword123!"
        )

        response = self.client.get(
            reverse(
                "conversation_detail",
                args=[self.conversation.id]
            )
        )

        self.assertEqual(response.status_code, 302)

    def test_customer_cannot_access_conversation_stream(self):
        self.client.login(
            username="customer1",
            password="TestPassword123!"
        )

        response = self.client.get(
            reverse(
                "conversation_stream",
                args=[self.conversation.id]
            )
        )

        self.assertEqual(response.status_code, 302)

    def test_staff_can_access_dashboard(self):
        self.client.login(
            username="staff1",
            password="TestPassword123!"
        )

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 200)
