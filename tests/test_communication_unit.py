"""Isolated envelope and ordering regressions; not native encryption evidence."""
import unittest
from unittest.mock import Mock

from src.communication import AgentCommunication, DeliveryError
from src.communication.communication_types import new_message, validate_message


class CommunicationUnitTests(unittest.TestCase):
    def setUp(self):
        self.journal = Mock()
        self.authorize = Mock(return_value=True)
        self.sanitize = Mock(side_effect=lambda record: record)
        self.deliver = Mock()
        self.client = AgentCommunication('sender', self.journal, self.authorize,
                                         self.sanitize, self.deliver)

    def send(self):
        return self.client.send('recipient', 'event', {'value': 'fixture'}, session_id='session')

    def test_exact_record_authorized_before_storage_and_delivery(self):
        order = []
        self.authorize.side_effect = lambda record: order.append(('authorize', record)) or True
        self.journal.append.side_effect = lambda record: order.append(('append', record))
        self.deliver.side_effect = lambda recipient, record: order.append(('deliver', record))
        message_id = self.send()
        self.assertEqual([x[0] for x in order], ['authorize', 'append', 'authorize', 'deliver'])
        self.assertEqual(order[0][1], order[1][1])
        self.assertEqual(order[1][1], order[2][1])
        self.assertEqual(order[2][1], order[3][1])
        self.assertEqual(order[0][1]['id'], message_id)

    def test_revocation_between_append_and_delivery(self):
        self.authorize.side_effect = [True, False]
        with self.assertRaises(DeliveryError) as error:
            self.send()
        self.assertIsInstance(error.exception.__cause__, PermissionError)
        self.assertEqual(error.exception.message_id, self.journal.append.call_args.args[0]['id'])
        self.deliver.assert_not_called()

    def test_policy_denial_and_sanitizer_identity_mutation(self):
        self.authorize.return_value = False
        with self.assertRaises(PermissionError):
            self.send()
        self.journal.append.assert_not_called()
        self.deliver.assert_not_called()
        self.authorize.reset_mock()
        self.sanitize.side_effect = lambda record: dict(record, sender='other')
        with self.assertRaises(PermissionError):
            self.send()
        self.authorize.assert_not_called()

    def test_storage_and_delivery_failure_distinguished(self):
        self.journal.append.side_effect = OSError('storage failure')
        with self.assertRaises(OSError):
            self.send()
        self.deliver.assert_not_called()
        self.journal.append.side_effect = None
        self.deliver.side_effect = OSError('delivery failure')
        with self.assertRaises(DeliveryError) as caught:
            self.send()
        self.assertEqual(caught.exception.message_id, self.journal.append.call_args.args[0]['id'])

    def test_nested_secret_and_nonfinite_payload_rejected(self):
        for payload in ({'nested': [{'api-key': 'fixture'}]}, {'value': float('nan')}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                new_message('sender', 'recipient', 'event', payload, session_id='session')

    def test_unknown_schema_fields_rejected(self):
        record = new_message('sender', 'recipient', 'event', {}, session_id='session')
        with self.assertRaises(ValueError):
            validate_message(dict(record, extra=True))


if __name__ == '__main__':
    unittest.main()
