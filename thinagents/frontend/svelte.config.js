import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			pages: '../web/ui/build',
			assets: '../web/ui/build',
			fallback: 'index.html'
		})
	}
};

export default config;
