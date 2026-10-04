"""Training-family-only few-shot examples; dev/test text must never enter this prompt."""

CATEGORY_EXAMPLES = {
    "broadband_outage": [
        "The optical box has a red LOS light and every device is offline. Restarting did nothing.",
        "The fibre lights are normal but neither my wired desktop nor Wi-Fi devices can browse. The router says authentication failed.",
    ],
    "intermittent_broadband": [
        "The connection drops on my wired computer too, mostly when it rains. Restarting the router did not help.",
        "Every evening the internet pauses on all devices, including my wired laptop. The optical light remains normal.",
    ],
    "slow_broadband": [
        "My upgrade to 300 Mbps completed, but repeated wired tests stop around the old 100 Mbps rate.",
        "Downloads on my Ethernet desktop are much slower than on another wired computer. My desktop link says 100 Mbps.",
    ],
    "wifi_connectivity": [
        "Wi-Fi is good beside the router but almost unusable upstairs. My wired computer is fine.",
        "Wi-Fi slows down in the apartment every evening but Ethernet stays fast, even at that time.",
    ],
    "router_ont_hardware": [
        "The router has no lights, but the wall socket works with a lamp.",
        "The router keeps restarting after its update. Internet was fine before the update began.",
    ],
    "billing_dispute": [
        "My statement shows two completed payments for the same invoice, not one pending card hold.",
        "An entertainment add-on appears on my bill, but I do not remember authorizing it.",
    ],
    "payment_restoration": [
        "My payment is settled in the portal, but my broadband still shows suspended for nonpayment.",
        "My bank shows a pending payment, while the provider still says unpaid and my service is paused.",
    ],
    "iptv": [
        "Live TV buffers over Wi-Fi but works when I connect the box by Ethernet.",
        "Only channels in my newly purchased TV package show not authorized. Existing channels play normally.",
    ],
    "mobile_coverage": [
        "Signal disappears inside my apartment but comes back outside. Another phone on the same network behaves the same way.",
        "Several neighbours and I suddenly have no mobile signal outdoors either.",
    ],
    "mobile_data": [
        "Calls and texts work after I changed phones, but mobile data does not. Wi-Fi works.",
        "Data became very slow after I used most of my allowance, while calls still work.",
    ],
    "voice_call_failure": [
        "Mobile data works, but every outgoing call fails. Incoming calls still arrive.",
        "Calls disconnect when I move between rooms, but data continues and stationary calls are better.",
    ],
    "number_porting": [
        "My number transfer was rejected because the details do not match the old provider's account.",
        "The transfer date is still tomorrow, but I expected the new SIM to work today. My old line still works.",
    ],
    "sim_esim_activation": [
        "My replacement physical SIM stays inactive. The serial on the order does not match the card I received.",
        "I scanned my eSIM code once on another phone and it now says already used on this one.",
    ],
    "sms_otp": [
        "Normal messages arrive but one bank's login codes never do. Other verification messages arrive.",
        "I can receive texts but cannot send any, even to ordinary mobile numbers.",
    ],
    "roaming": [
        "I am abroad with an active roaming pack; calls work but the account page says data roaming is disabled.",
        "My phone attached to a network abroad but has no data. Another listed partner is available.",
    ],
}
