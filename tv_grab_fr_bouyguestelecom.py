#!/usr/bin/python3

# Copyright 2020 Mohamed El Morabity
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

"""tv_grab_fr_bouyguestelecom.py - Grab French television listings using the
Bouygues Telecom PCTV API in XMLTV format.
"""

from argparse import ArgumentParser
from argparse import Namespace
from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta
import logging
import re
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Union

import dateutil.parser
import dateutil.tz
from lxml.etree import Element  # type: ignore # nosec
from lxml.etree import ElementTree  # nosec
from requests import Response
from requests import Session
from requests.exceptions import RequestException


class BouyguesTelecomException(Exception):
    """Base class for exceptions raised by the module."""


class BouyguesTelecom:
    """Implements grabbing and processing functionalities required to generate
    XMLTV data from Bouygues Telecom mobile API.
    """

    _API_BASE_URL = "https://www.bouyguestelecom.fr/tv-direct"
    _API_URL = f"{_API_BASE_URL}/data"
    _API_USER_AGENT = (
        "Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:81.0) "
        "Gecko/20100101 Firefox/81.0"
    )
    _API_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
    _API_XMLTV_CREDIT = {
        "Acteur": "actor",
    }
    _API_ETSI_CATEGORIES = {
        "Divertissement": "Show / Game show",
        "Documentaire": "News / Current affairs",
        "Film": "Movie / Drama",
        "Jeunesse": "Children's / Youth programmes",
        "Journal": "News / Current affairs",
        "Magazine": "Magazines / Reports / Documentary",
        "Musique": "Music / Ballet / Dance",
        "Sport": "Sports",
        "Série": "Movie / Drama",
        "Téléfilm": "Movie / Drama",
        "Téléréalité": "Show / Game show",
    }

    _API_TIMEZONE = dateutil.tz.gettz("Europe/Paris")
    _XMLTV_DATETIME_FORMAT = "%Y%m%d%H%M%S %z"

    def __init__(
        self,
        generator: Optional[str] = None,
        generator_url: Optional[str] = None,
    ):
        self._generator = generator
        self._generator_url = generator_url

        self._session = Session()
        self._session.headers.update({"User-Agent": self._API_USER_AGENT})
        self._session.hooks = {"response": [self._requests_raise_status]}

        self._channels = {
            self._bouyguestelecom_to_xmltv_id(key): {
                "id": key,
                "display-name": title,
                "icon": {"src": c.get("logoUrl")},
            }
            for c in self._query_api("list-chaines.json").get("body", [])
            if (key := c.get("key")) and (title := c.get("title"))
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._session:
            self._session.close()

    @staticmethod
    def _requests_raise_status(response: Response, *args, **kwargs) -> None:
        try:
            response.raise_for_status()
        except RequestException as ex:
            logging.debug(
                "Error while retrieving URL %s", response.request.url
            )
            try:
                raise BouyguesTelecomException(
                    response.json().get("message") or ex
                )
            except ValueError:
                raise BouyguesTelecomException(ex)

    def _query_api(
        self, path: str, **query: Union[int, str]
    ) -> Dict[str, Any]:
        url = "{}/{}".format(self._API_URL, path.strip("/"))
        response = self._session.get(url, params=query)

        logging.debug("Retrieved URL %s", response.request.url)

        return response.json()

    @classmethod
    def _bouyguestelecom_to_xmltv_id(cls, channel_id: int) -> str:

        return f"{channel_id}.bouyguestelecom.fr"

    def get_available_channels(self) -> Dict[str, str]:
        """Return the list of all available channels on Bouygues Telecom, with
        their XMLTV ID and name.
        """

        return {k: v["display-name"] for k, v in self._channels.items()}

    @staticmethod
    def _to_string(value: Union[None, bool, int, str]) -> Optional[str]:
        if isinstance(value, bool):
            return "yes" if value else "no"

        if value:
            stripped_value = str(value).strip()

        if not value or not stripped_value:
            return None

        return stripped_value

    @staticmethod
    def _xmltv_element(
        tag: str,
        text: Union[None, int, str] = None,
        parent: Element = None,
        **attributes: Union[None, int, str],
    ) -> Element:
        attributes = {
            k: w
            for k, v in attributes.items()
            if (w := BouyguesTelecom._to_string(v))
        }

        element = Element(tag, **attributes)
        element.text = BouyguesTelecom._to_string(text)

        if parent is not None:
            parent.append(element)

        return element

    @staticmethod
    def _xmltv_element_with_text(
        tag: str,
        text: Union[None, int, str],
        parent: Element = None,
        **attributes: Optional[str],
    ) -> Optional[Element]:
        if not text:
            return None

        return BouyguesTelecom._xmltv_element(
            tag, text=text, parent=parent, **attributes
        )

    def _to_xmltv_channel(self, channel_id: str) -> Optional[Element]:
        xmltv_channel = Element("channel", id=channel_id)

        channel_data = self._channels.get(channel_id)
        if not channel_data:
            return None

        # Channel display name
        self._xmltv_element_with_text(
            "display-name",
            channel_data.get("display-name"),
            parent=xmltv_channel,
        )

        # Icon associated to the programme
        self._xmltv_element(
            "icon", parent=xmltv_channel, **channel_data.get("icon", {})
        )

        return xmltv_channel

    @staticmethod
    def _get_xmltv_ns_episode_number(
        season: Optional[int], episode: Optional[int],
    ) -> Optional[str]:
        if not season and not episode:
            return None

        result = ""

        if season:
            result = f"{season - 1}"

        result += "."

        if episode:
            result += f"{episode - 1}"

        result += ".0/1"

        return result

    # pylint: disable=too-many-locals
    def _to_xmltv_program(
        self,
        program: Dict[str, Any],
        channel_id: str,
        start: datetime,
        stop: Optional[datetime],
    ) -> Optional[Element]:

        xmltv_program = self._xmltv_element(
            "programme",
            start=start.strftime(self._XMLTV_DATETIME_FORMAT),
            stop=stop.strftime(self._XMLTV_DATETIME_FORMAT) if stop else None,
            channel=channel_id,
        )

        # Programme title
        title = program.get("longTitle") or program.get("title")
        xmltv_title = self._xmltv_element_with_text(
            "title", title, parent=xmltv_program
        )
        if xmltv_title is None:
            return None

        # Description of the programme or episode
        self._xmltv_element_with_text(
            "desc", program.get("summary"), parent=xmltv_program
        )

        # Credits for the programme
        xmltv_credits = self._xmltv_element("credits")
        self._xmltv_element_with_text("director", program.get("realisateur"))

        for people in program.get("characters", []):
            function = people.get("function")
            credit = self._API_XMLTV_CREDIT.get(function)
            if not credit:
                if function:
                    logging.debug(
                        'No XMLTV credit defined for function "%s"', function,
                    )
                continue

            name = "{} {}".format(
                people.get("firstName", ""), people.get("lastName", "")
            ).strip()
            if not name:
                continue

            self._xmltv_element_with_text(
                credit,
                name,
                parent=xmltv_credits,
                role=people.get("role") if credit == "actor" else None,
            )

        if len(xmltv_credits) > 0:
            xmltv_program.append(xmltv_credits)

        # Date the programme or film was finished
        self._xmltv_element_with_text(
            "date", program.get("productionDate"), parent=xmltv_program,
        )

        # Type of programme
        genres = program.get("genre", [])
        for genre in program.get("genre", []):
            self._xmltv_element_with_text(
                "category", genre, parent=xmltv_program, lang="fr",
            )
        etsi_category = next(
            (c for g in genres if (c := self._API_ETSI_CATEGORIES.get(g))),
            None,
        )
        self._xmltv_element_with_text(
            "category", etsi_category, parent=xmltv_program, lang="en"
        )
        if genres and not etsi_category:
            logging.debug(
                'No ETSI category found for genre(s) "%s"', ", ".join(genres)
            )

        # True length of the programme
        self._xmltv_element_with_text(
            "length",
            program.get("duration"),
            parent=xmltv_program,
            units="seconds",
        )

        # Icon associated to the programme
        media_url = program.get("urlMedia")
        if media_url and not media_url.startswith("http"):
            media_url = f"{self._API_BASE_URL}/{media_url}"
        self._xmltv_element("icon", parent=xmltv_program, src=media_url)

        # Episode number
        season = None
        try:
            season = int(program.get("seasonNumber") or 0)
        except ValueError:
            pass
        episode = None
        try:
            episode = int(program.get("episodeNumber") or 0)
        except ValueError:
            pass

        self._xmltv_element_with_text(
            "episode-num",
            self._get_xmltv_ns_episode_number(season, episode),
            parent=xmltv_program,
            system="xmltv_ns",
        )

        # Star rating
        if rating := program.get("rating"):
            self._xmltv_element_with_text(
                "value",
                f"{rating}/5",
                parent=self._xmltv_element(
                    "star-rating",
                    parent=xmltv_program,
                    system="Bouygues Telecom",
                ),
            )

        return xmltv_program

    def _get_xmltv_programs(
        self, channel_ids: List[str], days: int, offset: int
    ) -> Generator[Element, None, None]:

        start = datetime.combine(
            date.today(), time(0)
        ).astimezone().astimezone(tz=self._API_TIMEZONE) + timedelta(
            days=offset
        )
        end = start + timedelta(days=days)

        for channel_id in channel_ids:
            bouygues_telecom_channel_id = self._channels.get(
                channel_id, {}
            ).get("id")
            if not bouygues_telecom_channel_id:
                continue
            programs = self._query_api(
                f"epg/{bouygues_telecom_channel_id}.json",
                d="{}{}{}".format(start.year, start.month - 1, start.day),
            ).get("programs", [])

            for program in programs:
                try:
                    program_start = dateutil.parser.isoparse(
                        program.get("fullStartTime")
                    ).replace(tzinfo=self._API_TIMEZONE)
                    # print(start, program_start)
                except ValueError:
                    continue

                program_end = None
                try:
                    program_end = dateutil.parser.isoparse(
                        program.get("fullEndTime")
                    ).replace(tzinfo=self._API_TIMEZONE)
                except ValueError:
                    pass

                if program_start >= end:
                    continue

                if program_end and program_end <= start:
                    continue

                yield self._to_xmltv_program(
                    program, channel_id, program_start, program_end
                )

    def _to_xmltv(
        self, channel_ids: List[str], days: int, offset: int
    ) -> ElementTree:
        xmltv = self._xmltv_element(
            "tv",
            **{
                "source-info-name": "Bouygues Telecom",
                "source-info-url": "https://www.bouyguestelecom.fr/tv-direct/",
                "source-data-url": self._API_URL,
                "generator-info-name": self._generator,
                "generator-info-url": self._generator_url,
            },
        )

        xmltv_channels = {}  # type: Dict[str, Element]
        xmltv_programs = []

        for xmltv_program in self._get_xmltv_programs(
            channel_ids, days, offset
        ):
            if xmltv_program is None:
                continue
            channel_id = xmltv_program.get("channel")
            if channel_id not in xmltv_channels:
                xmltv_channels[channel_id] = self._to_xmltv_channel(channel_id)
            xmltv_programs.append(xmltv_program)

        xmltv.extend(xmltv_channels.values())
        xmltv.extend(xmltv_programs)

        return ElementTree(xmltv)

    def write_xmltv(
        self, channel_ids: List[str], output_file: Path, days: int, offset: int
    ) -> None:
        """Grab Bouygues Telecom programs in XMLTV format and write them to
        file.
        """

        logging.debug("Writing XMLTV program to file %s", output_file)

        xmltv_data = self._to_xmltv(channel_ids, days, offset)
        xmltv_data.write(
            str(output_file),
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
        )


_PROGRAM = "tv_grab_fr_bouyguestelecom"
__version__ = "1.0"
__url__ = "https://github.com/melmorabity/tv_grab_fr_bouyguestelecom"

_DESCRIPTION = "France (Bouygues Telecom)"
_CAPABILITIES = ["baseline", "manualconfig"]

_DEFAULT_DAYS = 1
_DEFAULT_OFFSET = 0

_DEFAULT_CONFIG_FILE = Path.home().joinpath(".xmltv", f"{_PROGRAM}.conf")


def _print_description() -> None:
    print(_DESCRIPTION)


def _print_version() -> None:
    print("This is {} version {}".format(_PROGRAM, __version__))


def _print_capabilities() -> None:
    print("\n".join(_CAPABILITIES))


def _parse_cli_args() -> Namespace:
    parser = ArgumentParser(
        description="get French television listings using Bouygues Telecom "
        "mobile API in XMLTV format"
    )
    parser.add_argument(
        "--description",
        action="store_true",
        help="print the description for this grabber",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="show the version of this grabber",
    )
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help="show the capabilities this grabber supports",
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        help="generate the configuration file by asking the users which "
        "channels to grab",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=_DEFAULT_DAYS,
        help="grab DAYS days of TV data (default: %(default)s)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=_DEFAULT_OFFSET,
        help="grab TV data starting at OFFSET days in the future (default: "
        "%(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/dev/stdout"),
        help="write the XML data to OUTPUT instead of the standard output",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=_DEFAULT_CONFIG_FILE,
        help="file name to write/load the configuration to/from (default: "
        "%(default)s)",
    )

    log_level_group = parser.add_mutually_exclusive_group()
    log_level_group.add_argument(
        "--quiet",
        action="store_true",
        help="only print error-messages on STDERR",
    )
    log_level_group.add_argument(
        "--debug",
        action="store_true",
        help="provide more information on progress to stderr to help in"
        "debugging",
    )

    return parser.parse_args()


def _read_configuration(
    available_channels: Dict[str, str], config_file: Path
) -> List[str]:

    channel_ids = set()
    with config_file.open("r") as config_reader:
        for line in config_reader:
            match = re.search(r"^\s*channel\s*=\s*(.+)\s*$", line)
            if match is None:
                continue

            channel_id = match.group(1)
            if channel_id in available_channels:
                channel_ids.add(channel_id)

    return list(channel_ids)


def _write_configuration(channel_ids: List[str], config_file: Path) -> None:

    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as config:
        for channel_id in channel_ids:
            print("channel={}".format(channel_id), file=config)


def _configure(available_channels: Dict[str, str], config_file: Path) -> None:
    channel_ids = []
    answers = ["yes", "no", "all", "none"]
    select_all = False
    select_none = False
    print(
        "Select the channels that you want to receive data for.",
        file=sys.stderr,
    )
    for channel_id, channel_name in available_channels.items():
        if not select_all and not select_none:
            while True:
                prompt = f"{channel_name} [{answers} (default=no)] "
                answer = input(prompt).strip()  # nosec
                if answer in answers or answer == "":
                    break
                print(
                    f"invalid response, please choose one of {answers}",
                    file=sys.stderr,
                )
            select_all = answer == "all"
            select_none = answer == "none"
        if select_all or answer == "yes":
            channel_ids.append(channel_id)
        if select_all:
            print(f"{channel_name} yes", file=sys.stderr)
        elif select_none:
            print(f"{channel_name} no", file=sys.stderr)

    _write_configuration(channel_ids, config_file)


def _main() -> None:
    args = _parse_cli_args()

    if args.version:
        _print_version()
        sys.exit()

    if args.description:
        _print_description()
        sys.exit()

    if args.capabilities:
        _print_capabilities()
        sys.exit()

    logging_level = logging.INFO
    if args.quiet:
        logging_level = logging.ERROR
    elif args.debug:
        logging_level = logging.DEBUG
    logging.basicConfig(
        level=logging_level, format="%(levelname)s: %(message)s",
    )

    try:
        tele_loisirs = BouyguesTelecom(
            generator=_PROGRAM, generator_url=__url__
        )
    except BouyguesTelecomException as ex:
        logging.error(ex)
        sys.exit(1)

    logging.info("Using configuration file %s", args.config_file)

    available_channels = tele_loisirs.get_available_channels()
    if args.configure:
        _configure(available_channels, args.config_file)
        sys.exit()

    if not args.config_file.is_file():
        logging.error(
            "You need to configure the grabber by running it with --configure"
        )
        sys.exit(1)

    channel_ids = _read_configuration(available_channels, args.config_file)
    if not channel_ids:
        logging.error(
            "Configuration file %s is empty or malformed, delete and run with "
            "--configure",
            args.config_file,
        )
        sys.exit(1)

    try:
        tele_loisirs.write_xmltv(
            channel_ids, args.output, args.days, args.offset
        )
    except BouyguesTelecomException as ex:
        logging.error(ex)


if __name__ == "__main__":
    _main()
