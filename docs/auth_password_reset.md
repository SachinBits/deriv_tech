# Password Reset and Account Lockout

This article explains how password resets work and what happens after repeated failed sign-in attempts.

## Requesting a password reset

Users can request a password reset from the sign-in page by selecting "Forgot password" and entering the email address on the account. Users can request up to 3 password resets per hour. Additional requests within the same hour are rejected, and the user sees a message asking them to try again later.

## Reset link expiry

Each reset email contains a single-use link. Reset links expire after 30 minutes. If the link has expired, the user must request a new reset from the sign-in page. Requesting a new link invalidates any earlier link that has not been used.

## Account lockout after failed logins

To protect accounts from password guessing, the account locks for 15 minutes after 5 failed login attempts. During the lockout period, sign-in is blocked even if the correct password is entered. The lock clears automatically when the 15 minutes have passed, and no action from support is needed.

## Tips for support agents

Ask the user to check their spam folder before requesting another reset email. Confirm that the email address they entered matches the one registered on the account.
