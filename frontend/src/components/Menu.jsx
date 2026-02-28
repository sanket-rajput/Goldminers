'use client';

import { allCocktails } from '../../constants/index.js'
import { useRef, useState } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap';

const Menu = () => {
	const contentRef = useRef();
	const [currentIndex, setCurrentIndex] = useState(0);

	useGSAP(() => {
		gsap.fromTo('#title', { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 1, ease: 'power2.out' });
		gsap.fromTo('.cocktail img', { opacity: 0, scale: 0.8 }, {
			scale: 1, opacity: 1, duration: 1.2, ease: 'back.out(1.7)'
		})
		gsap.fromTo('.details h2', { yPercent: 100, opacity: 0 }, {
			yPercent: 0, opacity: 1, duration: 1, ease: 'power2.out'
		})
		gsap.fromTo('.details p', { yPercent: 50, opacity: 0 }, {
			yPercent: 0, opacity: 1, duration: 1, ease: 'power2.out', delay: 0.2
		})
	}, [currentIndex]);

	const totalCocktails = allCocktails.length;

	const goToSlide = (index) => {
		const newIndex = (index + totalCocktails) % totalCocktails;
		setCurrentIndex(newIndex);
	}

	const getCocktailAt = (indexOffset) => {
		return allCocktails[(currentIndex + indexOffset + totalCocktails) % totalCocktails]
	}

	const currentCocktail = getCocktailAt(0);

	return (
		<section id="menu" className="relative py-20 bg-surface overflow-hidden">
			<div className="noisy absolute inset-0 z-0 pointer-events-none"></div>

			<div className="container mx-auto px-5">
				<h2 className="text-center text-lavender uppercase tracking-widest text-sm font-semibold mb-6">Architecture Overview</h2>

				<div className="cocktail-tabs flex flex-wrap justify-center gap-2 md:gap-4 mb-10" aria-label="Feature Navigation">
					{allCocktails.map((cocktail, index) => {
						const isActive = index === currentIndex;
						return (
							<button
								key={cocktail.id}
								className={`px-4 md:px-6 py-2 rounded-full border transition-all duration-300 cursor-pointer whitespace-nowrap ${isActive
									? 'bg-sage border-sage text-bg-base font-medium shadow-[0_0_20px_var(--glow-sage)]'
									: 'border-light text-subdued hover:text-secondary hover:border-secondary'
									}`}
								onClick={() => goToSlide(index)}
							>
								{cocktail.name}
							</button>
						)
					})}
				</div>

				<div className="content relative flex flex-col items-center gap-10">
					<div className="recipe w-full lg:max-w-4xl backdrop-blur-xl bg-bg-card/30 p-8 md:p-12 rounded-[2rem] border border-light shadow-base">
						<div ref={contentRef} className="info mb-8">
							<p id="title" className="text-sage text-3xl md:text-4xl font-light mb-2">{currentCocktail.name}</p>
							<div className="w-12 h-1 bg-lavender rounded-full"></div>
						</div>

						<div className="details space-y-6">
							<h2 className="text-primary text-2xl md:text-3xl font-light leading-tight">{currentCocktail.title}</h2>
							<p className="text-secondary text-base md:text-lg font-light leading-relaxed">{currentCocktail.description}</p>

							<div className="pt-6">
								<a href="#cocktails" className="inline-flex items-center gap-2 text-aqua hover:text-sage transition-colors text-sm font-medium uppercase tracking-wider">
									Technical Documentation
									<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M7 7h10v10M7 17L17 7" /></svg>
								</a>
							</div>
						</div>
					</div>
				</div>
			</div>
		</section>
	)
}
export default Menu
