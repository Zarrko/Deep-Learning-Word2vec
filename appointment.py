import re


def is_appointment(notice):
    """
    :param notice: Notice class
    :return: True is the Notice is an appointment
    """

    has_appointment = re.findall(r'appointment', notice, re.I)
    if has_appointment:
        return True
    return False

