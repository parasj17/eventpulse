"""
Email delivery.

Sends real email over SMTP when a server is configured in the environment, and
otherwise falls back to a clearly-labelled simulation, so registration still
works end to end on a machine with no mail account.

    SMTP_HOST        mail server, e.g. smtp.gmail.com
    SMTP_PORT        default 587 (465 when SMTP_SECURITY=ssl)
    SMTP_USER        login, if the server needs one
    SMTP_PASSWORD    password or app password
    SMTP_FROM        the From address, e.g. "EventPulse <no-reply@yourdomain>"
    SMTP_SECURITY    starttls (default), ssl, or none

Or put the same lines (KEY=value) in backend/.mail, a private file the server reads when
the environment doesn't set them. For Gmail, two lines are enough:

    GMAIL_USER=you@gmail.com
    GMAIL_APP_PASSWORD=abcd efgh ijkl mnop     (Google account > Security > App passwords)

Send a test:  python -m engine.mailer you@example.com

Uses only the standard library. Notes for real use:

* Gmail and most providers need an app password, not the account password.
* Mail from an address the sending server is not authorised for may land in
  spam. Use a sender on a domain you have set SPF/DKIM up for.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage

TIMEOUT_SECONDS = 15
MAIL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mail")


def _cfg():
    """Mail settings: the environment first, then backend/.mail. A Gmail address and app
    password stand in for the full SMTP settings."""
    c = {}
    try:
        with open(MAIL_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    c[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_SECURITY",
              "GMAIL_USER", "GMAIL_APP_PASSWORD"):
        if os.environ.get(k):
            c[k] = os.environ[k]
    if c.get("GMAIL_USER") and c.get("GMAIL_APP_PASSWORD") and not c.get("SMTP_HOST"):
        c.update(SMTP_HOST="smtp.gmail.com", SMTP_PORT="587", SMTP_SECURITY="starttls", SMTP_USER=c["GMAIL_USER"],
                 SMTP_PASSWORD=c["GMAIL_APP_PASSWORD"].replace(" ", ""),
                 SMTP_FROM=c.get("SMTP_FROM") or f"EventPulse <{c['GMAIL_USER']}>")
    return c


def configured():
    c = _cfg()
    return bool(c.get("SMTP_HOST") and c.get("SMTP_FROM"))


def _simulate(to, subject, body):
    print(f"[email simulated -> {to}] {subject}\n{body}")
    return {"provider": "simulated", "status": "simulated", "delivered": False,
            "to": to, "subject": subject, "body": body}


def send_email(to, subject, body, html=None):
    """
    Send a plain-text email. Never raises: a failure comes back as
    `status == "failed"` with an `error` message, so a mail outage cannot take
    registration down with it.
    """
    if not configured():
        r = _simulate(to, subject, body)
        if html:
            r["html"] = html
        return r

    c = _cfg()
    result = {"provider": "smtp", "to": to, "subject": subject, "body": body}
    security = c.get("SMTP_SECURITY", "starttls").lower()
    host = c["SMTP_HOST"]
    port = int(c.get("SMTP_PORT") or (465 if security == "ssl" else 587))

    msg = EmailMessage()
    msg["From"] = c["SMTP_FROM"]
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")

    try:
        if security == "ssl":
            server = smtplib.SMTP_SSL(host, port, timeout=TIMEOUT_SECONDS, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(host, port, timeout=TIMEOUT_SECONDS)
        with server:
            server.ehlo()
            if security == "starttls":
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if c.get("SMTP_USER"):
                server.login(c["SMTP_USER"], c.get("SMTP_PASSWORD", ""))
            server.send_message(msg)
        result.update(status="sent", delivered=True)
    except smtplib.SMTPAuthenticationError:
        result.update(status="failed", delivered=False, error="The mail server rejected the login")
    except smtplib.SMTPRecipientsRefused:
        result.update(status="failed", delivered=False, error="The mail server refused that address")
    except Exception as e:  # network down, DNS, TLS, timeout
        result.update(status="failed", delivered=False,
                      error=f"Could not send the email ({type(e).__name__})")
    return result


if __name__ == "__main__":
    import sys
    to = sys.argv[1] if len(sys.argv) > 1 else None
    if not to:
        sys.exit("Usage: python -m engine.mailer you@example.com")
    print("configured:", configured())
    r = send_email(to, "EventPulse test email", "If you can read this, EventPulse can email attendees their live status link.")
    print({k: v for k, v in r.items() if k != "body"})
