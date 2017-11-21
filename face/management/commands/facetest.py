from django.conf import settings

from django.contrib.staticfiles.management.commands.runserver import (
    Command as StaticFilesRunserverCommand,
)


class Command(StaticFilesRunserverCommand):
    help = "Starts a lightweight Web server for development and also serves static files."

    def add_arguments(self, parser):
        super(Command, self).add_arguments(parser)
        parser.add_argument(
            '--nostatic', action="store_false", dest='use_static_handler', default=True,
            help='Tells Django to NOT automatically serve static files at STATIC_URL.',
        )
        parser.add_argument(
            '--insecure', action="store_true", dest='insecure_serving', default=False,
            help='Allows serving static files even if DEBUG is False.',
        )

    def get_handler(self, *args, **options):
        """
        Returns the static files serving handler wrapping the default handler,
        if static files should be served. Otherwise just returns the default
        handler.
        """
        handler = super(Command, self).get_handler(*args, **options)
        return handler
