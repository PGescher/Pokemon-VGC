import argparse

from TemplateCompetitor.my_competitor import MyCompetitor
from vgc2.net.server import RemoteCompetitorManager, BASE_PORT


def main(_args):
    _id = _args.id
    competitor = MyCompetitor(name=f"Example {_id}")
    server = RemoteCompetitorManager(competitor, port=BASE_PORT + _id, authkey=f'Competitor {_id}'.encode('utf-8'))
    print(f"[{competitor.name}] Listening on port {BASE_PORT + _id} with authkey 'Competitor {_id}'")
    server.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()
    main(args)
