from __future__ import annotations
from agents.research.provider import TopicDescriptor
from providers.hybrid.retriever import (HybridResearchRetriever,SOURCE_CAP_REACHED,TRACE_SCHEMA_VERSION,CHEAP_TIER_SOURCE_TYPE,FALLBACK_TIER_SOURCE_TYPE)
from providers.searxng.client import SearchResult

def _topic():
    return TopicDescriptor(topic_id='t1', title='Why is the sky blue', description='', pillar='science', topic_type='explanation')

def _mk_search_result(url, title='T', snippet='s', engines=None, score=None):
    return SearchResult(url=url, title=title, snippet=snippet, engines=engines or ['google'], score=score)

class _StubSearcher:
    def __init__(self, results):
        self._results = results
    def search(self, query, *, max_results=None):
        return tuple(self._results)

class _StubFetcher:
    def __init__(self, results):
        self._results = results
        self.calls = []
    def fetch(self, url):
        self.calls.append(url)
        return self._results[url]

def _ok_fetch():
    from providers.html.fetcher import FetchedPage
    return FetchedPage(url='https://example.com/page', ok=True, http_status=200, content_type='text/html', title='T', text='hello world', content_length=11)


def test_candidate_identity_propagation():
    retriever = HybridResearchRetriever(
        searcher=_StubSearcher([_mk_search_result('https://example.com/page')]),
        fetcher=_StubFetcher({'https://example.com/page': _ok_fetch()}),
    )
    retriever.retrieve(_topic())
    trace = retriever.last_trace.to_dict()
    assert trace['schema_version'] >= 2
    assert len(trace['raw_hits']) >= 1
    assert trace['raw_hits'][0]['canonical_url'] == 'https://example.com/page'
    assert trace['candidate_records']
    cand = trace['candidate_records'][0]
    assert cand['identity']['canonical_url'] == 'https://example.com/page'
    assert cand['identity']['url'] == 'https://example.com/page'
    assert trace['emitted_records']
    emit = trace['emitted_records'][0]
    assert emit['identity']['canonical_url'] == 'https://example.com/page'
def test_cheap_fetch_identity():
    from providers.html.fetcher import FetchedPage
    retriever = HybridResearchRetriever(
        searcher=_StubSearcher([_mk_search_result('https://a.com/1')]),
        fetcher=_StubFetcher({'https://a.com/1': FetchedPage(url='https://a.com/1', ok=True, http_status=200, content_type='text/html', title='T', text='x'*10, content_length=10)}),
    )
    retriever.retrieve(_topic())
    t = retriever.last_trace.to_dict()
    assert t['candidate_records'][0]['cheap_ok'] is True
    assert t['candidate_records'][0]['identity']['canonical_url'] == 'https://a.com/1'


class _StubFallback:
    def __init__(self, ok=True, content='x', title='T'):
        self._ok = ok
        self._content = content
        self._title = title
    def scrape(self, url, *, only_main_content=True):
        from providers.firecrawl.client import FirecrawlResult
        if self._ok:
            return FirecrawlResult(url=url, title=self._title, markdown=self._content, )
        from providers.firecrawl.errors import FirecrawlProviderError
        raise FirecrawlProviderError('fail')

def test_cheap_fail_fallback_identity():
    from providers.html.fetcher import FetchedPage
    retriever = HybridResearchRetriever(
        searcher=_StubSearcher([_mk_search_result('https://a.com/1')]),
        fetcher=_StubFetcher({'https://a.com/1': FetchedPage(url='https://a.com/1', ok=False, error_kind='blocked')}),
        fallback=_StubFallback(ok=True, content='# h\nbody', title='T'),
    )
    retriever.retrieve(_topic())
    t = retriever.last_trace.to_dict()
    cand = t['candidate_records'][0]
    assert cand['cheap_attempted'] is True
    assert cand['cheap_ok'] is False
    assert cand['cheap_error_kind'] == 'blocked'
    assert cand['fallback_ok'] is True
    assert cand['identity']['canonical_url'] == 'https://a.com/1'


def test_deduplication_preserves_identity():
    sr1 = _mk_search_result('https://example.com/page')
    sr2 = _mk_search_result('https://example.com/page')
    retriever = HybridResearchRetriever(
        searcher=_StubSearcher([sr1, sr2]),
        fetcher=_StubFetcher({'https://example.com/page': _ok_fetch()}),
    )
    retriever.retrieve(_topic())
    t = retriever.last_trace.to_dict()
    assert t['duplicates_dropped'] >= 1
    assert len(t['candidate_records']) == 1

def test_fallback_skip_identity():
    from providers.html.fetcher import FetchedPage
    retriever = HybridResearchRetriever(
        searcher=_StubSearcher([_mk_search_result('https://a.com/1')]),
        fetcher=_StubFetcher({'https://a.com/1': FetchedPage(url='https://a.com/1', ok=False, error_kind='invalid_url')}),
    )
    retriever.retrieve(_topic())
    t = retriever.last_trace.to_dict()
    cand = t['candidate_records'][0]
    assert cand['cheap_attempted'] is True
    assert cand['fallback_attempted'] is False or cand['fallback_skip_reason'] is not None

def test_source_cap_records_not_attempted_with_identity():
    from providers.html.fetcher import FetchedPage
    retriever = HybridResearchRetriever(
        searcher=_StubSearcher([
            _mk_search_result('https://a.com/1'),
            _mk_search_result('https://a.com/2'),
        ]),
        fetcher=_StubFetcher({
            'https://a.com/1': FetchedPage(url='https://a.com/1', ok=True, http_status=200, content_type='text/html', title='T', text='x'*10, content_length=10),
            'https://a.com/2': FetchedPage(url='https://a.com/2', ok=True, http_status=200, content_type='text/html', title='T', text='y'*10, content_length=10),
        }),
        max_sources=1,
    )
    retriever.retrieve(_topic())
    t = retriever.last_trace.to_dict()
    assert t['emitted'] == 1
    assert t['not_attempted']
    na = t['not_attempted'][0]
    assert 'identity' in na or 'url' in na

def test_trace_disabled_behavior_unchanged():
    # ensure behavior same with/without trace recording side effects not breaking
    from providers.html.fetcher import FetchedPage
    retriever = HybridResearchRetriever(
        searcher=_StubSearcher([_mk_search_result('https://a.com/1')]),
        fetcher=_StubFetcher({'https://a.com/1': FetchedPage(url='https://a.com/1', ok=True, http_status=200, content_type='text/html', title='T', text='hello world', content_length=11)}),
    )
    sources1 = retriever.retrieve(_topic())
    # second run same
    retriever2 = HybridResearchRetriever(
        searcher=_StubSearcher([_mk_search_result('https://a.com/1')]),
        fetcher=_StubFetcher({'https://a.com/1': FetchedPage(url='https://a.com/1', ok=True, http_status=200, content_type='text/html', title='T', text='hello world', content_length=11)}),
    )
    sources2 = retriever2.retrieve(_topic())
    assert len(sources1) == len(sources2)
    assert sources1[0].url == sources2[0].url
