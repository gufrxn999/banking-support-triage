"""Domain taxonomy: which intents trigger which action.

The classifier predicts one of BANKING77's 77 label *ids*; this module decides
what that label means for routing. Keeping the mapping here (rather than inline
`if intent == ...` checks scattered through the graph) means a reviewer can
audit routing policy in one screen.

A note on the dataset: two label names are malformed upstream
(`Refund_not_showing_up`, `reverted_card_payment?`). We always route on the
integer id and treat the name as display-only.
"""

from __future__ import annotations

# Intents where a live exchange-rate lookup is the right answer.
# Only `exchange_rate` is a genuine "what is the rate" question.
LOOKUP_INTENTS: frozenset[str] = frozenset({"exchange_rate"})

# Currency-adjacent intents. These mention exchange rates but are complaints or
# policy questions; answering them with a quote would be wrong (and, for the
# dispute intents, actively harmful). They go to the FAQ path, never the tool.
CURRENCY_DISPUTE_INTENTS: frozenset[str] = frozenset(
    {
        "card_payment_wrong_exchange_rate",
        "wrong_exchange_rate_for_cash_withdrawal",
    }
)

CURRENCY_INFO_INTENTS: frozenset[str] = frozenset(
    {
        "exchange_charge",
        "exchange_via_app",
        "fiat_currency_support",
        "supported_cards_and_currencies",
    }
)

CURRENCY_RELATED_INTENTS: frozenset[str] = (
    LOOKUP_INTENTS | CURRENCY_DISPUTE_INTENTS | CURRENCY_INFO_INTENTS
)

# Curated FAQ coverage for the first release. Every intent listed here has a
# hand-written answer; every intent *not* listed falls through to a generic
# acknowledgement plus a human handoff. Deliberately not all 77: a wrong
# confident answer is worse than a handoff.
FAQ_COVERED_INTENTS: frozenset[str] = frozenset(
    {
        # cards
        "card_arrival",
        "card_not_working",
        "lost_or_stolen_card",
        "card_about_to_expire",
        "change_pin",
        "pin_blocked",
        "compromised_card",
        "declined_card_payment",
        "card_payment_not_recognised",
        "transaction_charged_twice",
        # top-ups
        "top_up_limits",
        "top_up_failed",
        "pending_top_up",
        "automatic_top_up",
        "top_up_by_card_charge",
        # transfers
        "transfer_timing",
        "pending_transfer",
        "failed_transfer",
        "cancel_transfer",
        "transfer_not_received_by_recipient",
        "transfer_fee_charged",
        # money in
        "receiving_money",
        "transfer_into_account",
        "balance_not_updated_after_bank_transfer",
        # cash / atm
        "atm_support",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        # currency
        *sorted(CURRENCY_RELATED_INTENTS),
        # account
        "verify_my_identity",
        "why_verify_identity",
        "edit_personal_details",
        "terminate_account",
        "country_support",
        # Added after M1: these are the neighbour each of these intents was most
        # often confused with, so a wrong-but-confident route still lands on a
        # real answer instead of a generic handoff.
        "card_delivery_estimate",
        "virtual_card_not_working",
        "top_up_reverted",
    }
)

# Intents that must never be resolved automatically: they assert a dispute or a
# loss, so a human has to confirm facts before anyone promises money back.
NEVER_AUTO_ANSWER_INTENTS: frozenset[str] = frozenset(
    {
        "compromised_card",
        "lost_or_stolen_card",
        "transaction_charged_twice",
        "card_payment_not_recognised",
        "cash_withdrawal_not_recognised",
        "request_refund",
        "Refund_not_showing_up",
        "reverted_card_payment?",
        "declined_transfer",
        "wrong_amount_of_cash_received",
        *CURRENCY_DISPUTE_INTENTS,
    }
)


def display_name(label_name: str) -> str:
    """Turn a dataset label into something presentable to a user.

    `Refund_not_showing_up` -> "Refund not showing up"
    `reverted_card_payment?` -> "Reverted card payment"
    """
    cleaned = label_name.replace("_", " ").strip().rstrip("?").strip()
    return cleaned[:1].upper() + cleaned[1:]
