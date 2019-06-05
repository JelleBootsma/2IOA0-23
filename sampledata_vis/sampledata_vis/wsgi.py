"""
WSGI config for sampledata_vis project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os
from tornado.ioloop import IOLoop
from bokeh.server.server import Server
import application.Visualisations
from threading import Thread
from django.core.wsgi import get_wsgi_application
def serverWorker():
    server = Server(
        {'/Weighted' : application.Visualisations.Weighted,
         '/Grouped' : application.Visualisations.Grouped,
         '/Adjacent' : application.Visualisations.Adjacent},  # list of Bokeh applications
        io_loop=IOLoop(),        # Tornado IOLoop
        allow_websocket_origin=["127.0.0.1:8000", "localhost:5006"]
    )

    # start timers and services and immediately return
    server.start()
    server.io_loop.start()
Thread(target=serverWorker).start()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sampledata_vis.settings')

application = get_wsgi_application()
