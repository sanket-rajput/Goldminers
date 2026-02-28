import { useGSAP } from '@gsap/react'
import gsap from 'gsap';
import { cocktailLists, mockTailLists } from '../../constants/index.js'

const Cocktails = () => {
	useGSAP(() => {
		const parallaxTimeline = gsap.timeline({
			scrollTrigger: {
				trigger: '#cocktails',
				start: 'top 30%',
				end: 'bottom 80%',
				scrub: true,
			}
		})

		parallaxTimeline
			.from('#c-left-leaf', {
				x: -100, y: 100, opacity: 0
			})
			.from('#c-right-leaf', {
				x: 100, y: 100, opacity: 0
			})
	})

	return (
		<section id="cocktails" className="relative overflow-hidden bg-surface py-20">
			<div className="noisy absolute inset-0 z-0 pointer-events-none"></div>
			<img src="/images/cocktail-left-leaf.png" alt="l-leaf" id="c-left-leaf" className="opacity-20" />
			<img src="/images/cocktail-right-leaf.png" alt="r-leaf" id="c-right-leaf" className="opacity-20" />

			<div className="list container mx-auto relative z-10 px-5">
				<div className="popular">
					<h2 className="text-lavender uppercase tracking-widest text-sm font-semibold mb-8">Clinical Workflow</h2>

					<ul className="space-y-6">
						{cocktailLists.map(({ name, country, detail, price }) => (
							<li key={name} className="border-b border-light pb-4 group">
								<div className="md:me-28">
									<h3 className="text-sage text-2xl font-light group-hover:text-aqua transition-colors">{name}</h3>
									<p className="text-subdued text-xs mt-1 uppercase tracking-wider">{country} | {detail}</p>
								</div>
								<span className="text-secondary font-medium">{price}</span>
							</li>
						))}
					</ul>
				</div>

				<div className="loved">
					<h2 className="text-lavender uppercase tracking-widest text-sm font-semibold mb-8">Core Intelligence</h2>

					<ul className="space-y-6">
						{mockTailLists.map(({ name, country, detail, price }) => (
							<li key={name} className="border-b border-light pb-4 group">
								<div className="me-28">
									<h3 className="text-sage text-2xl font-light group-hover:text-aqua transition-colors">{name}</h3>
									<p className="text-subdued text-xs mt-1 uppercase tracking-wider">{country} | {detail}</p>
								</div>
								<span className="text-secondary font-medium">{price}</span>
							</li>
						))}
					</ul>
				</div>
			</div>
		</section>
	)
}

export default Cocktails
